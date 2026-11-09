import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    t0 = time.time()
    deadline = t0 + 135
    
    # Target tensor
    T = np.zeros((9, 18, 18))
    for i in range(3):
        for j in range(3):
            for k in range(6):
                T[i*3+j, j*6+k, k*3+i] = 1.0
    
    # Schoolbook
    def sb():
        R = 54
        U = np.zeros((9, R)); V = np.zeros((18, R)); W = np.zeros((18, R))
        c = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U[i*3+j, c] = 1; V[j*6+k, c] = 1; W[k*3+i, c] = 1; c += 1
        return U, V, W
    
    bU, bV, bW = sb()
    bestR = 54
    
    def kr(A, B):
        return (A[:,:,None] * B[None,:,:]).T.reshape(-1, A.shape[1])
    
    def rec(U, V, W):
        return sum(np.outer(np.outer(U[:,r], V[:,r]), W[:,r]) for r in range(U.shape[1]))
    
    for R in range(35, 54):
        if time.time() > deadline - 10: break
        for tr in range(15):
            if time.time() > deadline - 5: break
            np.random.seed(R*100+tr)
            
            # Init: take schoolbook columns with random swaps
            U0, V0, W0 = sb()
            perm = np.random.permutation(54)
            U = U0[:, perm[:R]].copy()
            V = V0[:, perm[:R]].copy()
            W = W0[:, perm[:R]].copy()
            
            # ALS
            for it in range(80):
                if time.time() > deadline - 5: break
                try:
                    U = T.reshape(9,-1) @ np.linalg.pinv(kr(V,W).T)
                    V = T.reshape(18,-1) @ np.linalg.pinv(kr(U,W).T)
                    W = T.reshape(18,-1) @ np.linalg.pinv(kr(U,V).T)
                except: break
            
            # Round
            Ur = np.round(U*2)/2; Vr = np.round(V*2)/2; Wr = np.round(W*2)/2
            
            # Check
            if np.allclose(rec(Ur,Vr,Wr), T, atol=1e-10):
                bU,bV,bW = Ur,Vr,Wr; bestR = R; break
        if bestR == R: break
    
    return {"U": [[float(x) for x in r] for r in bU],
            "V": [[float(x) for x in r] for r in bV],
            "W": [[float(x) for x in r] for r in bW]}
