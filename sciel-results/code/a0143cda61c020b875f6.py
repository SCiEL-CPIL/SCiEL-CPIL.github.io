import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    nm, mp, pn = n*m, m*p, p*n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Precompute flat tensors
    T1 = T.reshape(nm, mp*pn)  # for U
    T2 = T.reshape(mp, nm*pn)  # for V
    T3 = T.reshape(pn, nm*mp)  # for W
    
    def check(U, V, W, r):
        # Fast check using einsum
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)
    
    # Schoolbook
    best_r = nm * mp // pn * pn  # n*m*p
    best_r = n*m*p
    U_sb = np.zeros((nm, best_r))
    V_sb = np.zeros((mp, best_r))
    W_sb = np.zeros((pn, best_r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def als_decompose(r, n_iter=100, seed=0):
        if time.time() > deadline:
            return None
        np.random.seed(seed)
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        for it in range(n_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            M = np.einsum('jc,kc->cjk', V, W).reshape(r, -1)
            U = np.linalg.lstsq(M.T, T1.T, rcond=None)[0].T
            
            # Update V
            M = np.einsum('ic,kc->cik', U, W).reshape(r, -1)
            # T2[j, i*pn+k] = sum_c V[j,c] * U[i,c] * W[k,c]
            # M[c, i*pn+k] = U[i,c] * W[k,c]
            M = np.einsum('ic,kc->cik', U, W).reshape(r, -1)
            V = np.linalg.lstsq(M.T, T2.T, rcond=None)[0].T
            
            # Update W
            M = np.einsum('ic,jc->cij', U, V).reshape(r, -1)
            W = np.linalg.lstsq(M.T, T3.T, rcond=None)[0].T
        
        if check(U, V, W, r):
            return U, V, W
        
        # Try rounding
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if check(U_r, V_r, W_r, r):
                return U_r, V_r, W_r
        return None
    
    # Try ranks
    for r in range(40, best_r):
        if time.time() > deadline:
            break
        for seed in range(5):
            if time.time() > deadline:
                break
            result = als_decompose(r, n_iter=100, seed=10000 + r*100 + seed)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_r = r
                break
        if best_r == r:
            break
    
    U_list = [[float(best_U[i,c]) for c in range(best_r)] for i in range(nm)]
    V_list = [[float(best_V[i,c]) for c in range(best_r)] for i in range(mp)]
    W_list = [[float(best_W[i,c]) for c in range(best_r)] for i in range(pn)]
    return {"U": U_list, "V": V_list, "W": W_list}
