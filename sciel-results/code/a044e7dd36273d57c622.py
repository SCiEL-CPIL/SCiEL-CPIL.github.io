import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 8
    nm, mp, pn = n*m, m*p, p*n  # 16, 32, 32
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_result = None
    best_rank = nm * mp * pn
    
    for R in range(32, 129):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(R)
        U = np.random.randn(nm, R) * 0.1
        V = np.random.randn(mp, R) * 0.1
        W = np.random.randn(pn, R) * 0.1
        
        for it in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U: U @ C_2d = T_flat where C_2d[r, (beta,gamma)] = V[beta,r]*W[gamma,r]
            C_2d = np.einsum('br,gr->rg', V, W).T  # C_2d[r, beta*pn+gamma]... 
            # Actually: C_2d has shape (R, mp*pn) where C_2d[r, beta*pn+gamma] = V[beta,r]*W[gamma,r]
            # V is (mp, R), W is (pn, R)
            # V[beta, r] * W[gamma, r] for all beta, gamma, r
            # This is: (V.T * W.T).T reshaped
            C_2d = (V.T * W.T).T.reshape(R, mp * pn)  # (R, mp*pn)
            # U @ C_2d = T_flat (nm, mp*pn)
            T_flat = T.reshape(nm, mp * pn)
            U = np.linalg.lstsq(C_2d.T, T_flat.T, rcond=None)[0].T
            
            # Update V: V @ D_2d = T_flat_V where D_2d[r, (alpha,gamma)] = U[alpha,r]*W[gamma,r]
            D_2d = (U.T * W.T).T.reshape(R, nm * pn)
            T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            V = np.linalg.lstsq(D_2d.T, T_flat_V.T, rcond=None)[0].T
            
            # Update W: W @ E_2d = T_flat_W where E_2d[r, (alpha,beta)] = U[alpha,r]*V[beta,r]
            E_2d = (U.T * V.T).T.reshape(R, nm * mp)
            T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            W = np.linalg.lstsq(E_2d.T, T_flat_W.T, rcond=None)[0].T
            
            # Check error every 20 iterations
            if it % 20 == 19:
                # T_recon[alpha, beta, gamma] = sum_r U[alpha,r]*V[beta,r]*W[gamma,r]
                # = einsum('ar,br,gr->abg', U, V, W)
                T_recon = np.einsum('ar,br,gr->abg', U, V, W)
                error = np.sum((T_recon - T)**2)
                if error < 1e-10:
                    best_rank = R
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
        
        if best_result is not None:
            break
    
    if best_result is None:
        R = nm * mp * pn
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
