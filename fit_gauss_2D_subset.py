import numpy as np
import scipy as sp
from scipy import optimize
import pandas as pd
import os
import sys
import gc
import tqdm
import copy
#import gridData

#import sklearn as skl
#from sklearn.decomposition import PCA
#from sklearn.cluster import OPTICS, cluster_optics_dbscan

sysname=sys.argv[1]
leafnum=sys.argv[2]

print('reading headgroup data')
headgroupData=pd.read_csv('headgroup_data_with_leaflet_labels.csv')
print('grabbing subset for {sysname} system, leaflet {leafnum}'.format(
    sysname=sysname,leafnum=leafnum))
testData=headgroupData.query('(System == "{sysname}") and (Leaflet == {leafnum})'.format(
    sysname=sysname,leafnum=leafnum)).copy()
headgroupData=[]
gc.collect()

#gauss_params:
# zo: base height of gaussian
# h: scaling factor
# MUx, MUy: mean (center) of gaussian in x,y
# Sxx, Sxy, Syx, Syy: Covariance matrix of gaussian
#
# g2d(X) = z0 + h*exp((X-MU).T * S^-1 * (X-MU))
#
#X is a n_sample x 2 array
def gauss_2D(X,gauss_params,verbose=False):
    zo,h,mux,muy,sxx,sxy,syx,syy=gauss_params
    mu=np.array([mux,muy])
    sigInv=np.linalg.inv(np.matrix(np.array([[sxx,sxy],[syx,syy]])))
    if verbose:
        print(sigInv)
    return(
        np.array(np.apply_along_axis(
            lambda x: zo+h*np.exp(-np.abs(np.matrix(x-mu)*sigInv*np.matrix((x-mu)).T)),
            axis=1,arr=X)).flatten())
    
#takes a 2D gaussian specified by standard deviation on two principal axes
#and an angle of rotation to rotate the resulting ovoid definition in plane.
#enforces the gaussian to be concave up (bowl shape) by default
#can be changed to concave up or all either using concavity term
#concav='down' (default),'up','free'
#pax_params: [zo,h,sig_x,sig_y,theta]
#zo, h: base height and gaussian height / depth
#mux,muy: x and y coordinates of maximum height / depth respectively
#sig_x,sig_y: the standard deviation along the first and second principal axes
#theta: rotation angle in radians
def gauss_pAx_form(X,pax_params,concav='down',verbose=False):
    zo,h,mux,muy,sig_x,sig_y,theta=pax_params
    
    if concav=='down':
        H=-np.abs(h) #enforce bowl shape 
    elif concav=='up':
        H=np.abs(h) #enforce hill shape
    else:
        H=h #allow free fitting of concavity
    
    B0=np.matrix([
        [sig_x,0],
        [0,sig_y]
    ])

    tempRmat=np.matrix([
        [np.cos(theta),-np.sin(theta)],
        [np.sin(theta),np.cos(theta)]
    ])
    
    Brot=tempRmat*B0

    #normalized
    Bhat=np.matrix([
        (np.array(Brot[:,0])/np.sum(np.array(Brot[:,0])**2)).flatten(),
        (np.array(Brot[:,1])/np.sum(np.array(Brot[:,1])**2)).flatten()
    ]).T

    Amat=Bhat*np.linalg.inv(Brot)

    AmatInv=np.linalg.inv(Amat)
    
    if verbose:
        print('B0\n',B0)
        print('Brot\n',Brot)
        print('Bhat\n',Bhat)
        print('Amat\n',Amat)
        print('AmatInv\n',AmatInv)
        
    return(gauss_2D(X,np.array([zo,H,mux,muy,AmatInv[0,0],AmatInv[0,1],AmatInv[1,0],AmatInv[1,1]])))

#generates a function that takes pAx form parameterization
#and returns the root mean square deviation of the gaussian model
#over X from the corresponding values in y
#X should be an n_sample x 2 array, y should be an n_sample array
def get_gauss_pAx_score_function(X,y,giveDeltaVec=False):
    if giveDeltaVec:
        return(
            lambda pax_params: (
                np.sqrt(np.sum((y-gauss_pAx_form(X,pax_params))**2)),
                (y-gauss_pAx_form(X,pax_params))
            )
        )
    else:
        return(
            lambda pax_params: np.sqrt(np.sum((y-gauss_pAx_form(X,pax_params))**2)))

#same but for the original covariance matrix based gauss_2D model
def get_gauss_2D_score_function(X,y):
    return(
        lambda gauss_params: np.sqrt(np.sum((y-gauss_2D(X,gauss_params)**2))))

def get_pAx_initial_guess(X,Val,concavity='up',qcut=.05):
    if concavity=='up':
        zo=np.quantile(Val,q=1.-qcut)
        zClip=np.clip(Val,
                      np.quantile(Val,q=qcut),
                      zo)
        h=np.min(zClip)-np.max(zClip)
    elif concavity=='down':
        zo=np.quantile(Val,q=qcut)
        zClip=np.clip(Val,
                      zo,
                      np.quantile(Val,q=1.-qcut))
        h=np.max(zClip)-np.min(zClip)
    
    zDelta=np.abs(zClip-zo)
    zNorm=np.sum(zDelta)
    zFactor=zDelta/zNorm
    
    mux=np.sum(X[:,0]*zFactor)
    muy=np.sum(X[:,1]*zFactor)
    
    muVec=np.array([np.sum(X[:,ii]*zFactor) for ii in np.arange(X.shape[1])])
    
    Imat=np.matrix([
        [np.sum((X[:,iAx]-muVec[iAx])*(X[:,jAx]-muVec[jAx])*zFactor) \
         for jAx in np.arange(X.shape[1])] \
        for iAx in np.arange(X.shape[1])
    ])
    
    Ieig=np.linalg.eig(Imat)
    std_vec=np.sqrt(Ieig[0])
    
    std_mat=np.matrix(np.identity(len(std_vec))*std_vec)
    qmat=np.matrix(np.array([
        std_vec[ii]*Ieig[1][:,ii].flatten() \
        for ii in np.arange(len(std_vec))
    ]).T)
    
    Rmat=std_mat*np.linalg.inv(qmat)
    
    theta=np.mean(np.arccos(np.diag(Rmat)))
    
    mux,muy=muVec[:2]
    std_x,std_y=std_vec[:2]
    return(np.array([zo,h,mux,muy,std_x,std_y,theta]))

maxIters=1000
fitData=[]
with tqdm.tqdm(testData.groupby(['System','Frame','Leaflet'])) as groupPbar:
    with tqdm.tqdm() as fitPbar:
        for testGroup in groupPbar:
            testName,testSet=testGroup
            groupPbar.set_description_str(str(testName))
            sysName,frame,leaflet=testName

            def update_pbar(xk):
                fitPbar.set_description_str(','.join(list(map(lambda x:'%.1e'%x,xk))))
                fitPbar.update()

            testParmFit=sp.optimize.minimize(
                get_gauss_pAx_score_function(testSet[['X','Y']].to_numpy(),testSet['Z'].to_numpy()),
                get_pAx_initial_guess(testSet[['X','Y']].to_numpy(),testSet['Z'].to_numpy()),
                callback=update_pbar,
                method='BFGS',
                options={'maxiter':maxIters})

            fitEntry=[sysName,frame,leaflet]
            for val in testParmFit.x:
                fitEntry.append(val)

            fitData.append(copy.deepcopy(fitEntry))

            fitPbar.n=0
            fitPbar.refresh()

print("Exporting fit data")
fitData=np.array(fitData)
fitFrame=pd.DataFrame({
    'System':fitData[:,0],
    'Frame':fitData[:,1],
    'Leaflet':fitData[:,2],
    'zo':fitData[:,3],
    'h':fitData[:,4],
    'mux':fitData[:,5],
    'muy':fitData[:,6],
    'std_p1':fitData[:,7],
    'std_p2':fitData[:,8],
    'theta':fitData[:,9]
})
fitFrame.head()
fitFrame.to_csv('Gauss_2D_Fit_Data.{sysname}.leaflet_{leafnum}.csv'.format(
    sysname=sysname,leafnum=leafnum),index=False)
print("done")
