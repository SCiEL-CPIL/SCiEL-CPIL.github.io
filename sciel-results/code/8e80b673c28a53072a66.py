import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 8
    nm, mp, pn = n*m, m*p, p*n  # 6, 24, 16
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best = (U, V, W)
    
    for rank in range(24, rank_sb):
        if time.time() > deadline - 10:
            break
        
        success = False
        for trial in range(2):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            U_try = np.random.randn(nm, rank) * 0.1
            V_try = np.random.randn(mp, rank) * 0.1
            W_try = np.random.randn(pn, rank) * 0.1
            
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each a, solve least squares
                # A_mat[a] has shape (mp*pn, rank), A_mat[a][b*pn+c, t] = V[b,t]*W[c,t]
                # Vectorize: A_mat_all has shape (nm, mp*pn, rank)
                # A_mat_all[a, b*pn+c, t] = V[b,t] * W[c,t]
                VW = np.einsum('bt,ct->bct', V_try, W_try)  # (mp, pn, rank)
                VW_flat = VW.reshape(mp*pn, rank)  # (mp*pn, rank)
                # For each a, the RHS is T[a].flatten()
                # We can solve all a at once if we stack
                # But the design matrix is the same for all a!
                # So: T_flat = U @ VW_flat^T where T_flat has shape (nm, mp*pn)
                T_flat = T.reshape(nm, mp*pn)  # (nm, mp*pn)
                # T_flat[a, idx] = sum_t U[a,t] * VW_flat[idx, t]
                # So: T_flat = U @ VW_flat.T
                # U = T_flat @ pinv(VW_flat.T)
                try:
                    U_try = T_flat @ np.linalg.pinv(VW_flat.T)
                except:
                    pass
                
                # Update V
                UW = np.einsum('at,ct->act', U_try, W_try)  # (nm, pn, rank)
                UW_flat = UW.reshape(nm*pn, rank)
                T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm*pn)  # (mp, nm*pn)
                try:
                    V_try = T_flat_v @ np.linalg.pinv(UW_flat.T)
                except:
                    pass
                
                # Update W
                UV = np.einsum('at,bt->abt', U_try, V_try)  # (nm, mp, rank)
                UV_flat = UV.reshape(nm*mp, rank)
                T_flat_w = T.transpose(2, 0, 1).reshape(pn, nm*mp)  # (pn, nm*mp)
                try:
                    W_try = T_flat_w @ np.linalg.pinv(UV_flat.T)
                except:
                    pass
                
                # Check error
                pred = np.einsum('at,bt,ct->abc', U_try, V_try, W_try)
                err = np.sum(np.abs(pred - T))
                
                if err < 1e-8:
                    success = True
                    break
            
            if success:
                best = (U_try, V_try, W_try)
                break
        
        if success:
            break
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
