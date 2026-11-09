import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    T_flat_nm = T.reshape(nm, mp*pn)
    T_flat_mp = np.transpose(T, (1, 0, 2)).reshape(mp, nm*pn)
    T_flat_pn = np.transpose(T, (2, 0, 1)).reshape(pn, nm*mp)
    
    def try_rank(r, seed=0, max_iter=500, tol=1e-10):
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r) * 0.1
        V = rng.randn(mp, r) * 0.1
        W = rng.randn(pn, r) * 0.1
        
        for it in range(max_iter):
            M_U = np.einsum('br,cr->bc r', V, W).reshape(r, mp*pn)
            try:
                U = T_flat_nm @ M_U @ np.linalg.inv(M_U.T @ M_U)
            except:
                pass
            
            M_V = np.einsum('ar,cr->ac r', U, W).reshape(r, nm*pn)
            try:
                V = T_flat_mp @ M_V @ np.linalg.inv(M_V.T @ M_V)
            except:
                pass
            
            M_W = np.einsum('ar,br->ab r', U, V).reshape(r, nm*mp)
            try:
                W = T_flat_pn @ M_W @ np.linalg.inv(M_W.T @ M_W)
            except:
                pass
            
            if it % 50 == 0:
                reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.sum((reconstructed - T)**2)
                if err < tol:
                    break
        
        reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((reconstructed - T)**2)
        return U, V, W, err
    
    # Schoolbook
    rank_school = n * m * p
    U_s = np.zeros((nm, rank_school))
    V_s = np.zeros((mp, rank_school))
    W_s = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_s, V_s, W_s
    
    start_time = time.time()
    
    for r in [60, 64, 68, 70, 72]:
        if time.time() - start_time > 120:
            break
        for seed in range(5):
            if time.time() - start_time > 120:
                break
            U_try, V_try, W_try, err = try_rank(r, seed=seed, max_iter=300)
            if err < 1e-6:
                U_round = np.round(U_try * 2) / 2
                V_round = np.round(V_try * 2) / 2
                W_round = np.round(W_try * 2) / 2
                reconstructed = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                if np.max(np.abs(reconstructed - T)) < 1e-9:
                    best_rank = r
                    best_U, best_V, best_W = U_round, V_round, W_round
                    break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
