import numpy as np
import time

def construct():
    n, m, p = 3, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 12, 20, 15
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    deadline = time.time() + 105
    
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    for target_rank in range(best_rank - 1, 19, -1):
        if time.time() > deadline - 15:
            break
        
        found = None
        max_trials = 30
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 7919 + target_rank * 131 + 42)
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            for it in range(120):
                if time.time() > deadline - 5:
                    break
                
                for a in range(nm):
                    A = np.einsum('rb,rc->bcr', V, W).reshape(mp*pn, target_rank)
                    u_a, _, _, _ = np.linalg.lstsq(A, T[a].flatten(), rcond=None)
                    U[a] = u_a
                
                for b in range(mp):
                    A = np.einsum('ra,rc->acr', U, W).reshape(nm*pn, target_rank)
                    v_b, _, _, _ = np.linalg.lstsq(A, T[:, b, :].flatten(), rcond=None)
                    V[b] = v_b
                
                for c in range(pn):
                    A = np.einsum('ra,rb->abr', U, V).reshape(nm*mp, target_rank)
                    w_c, _, _, _ = np.linalg.lstsq(A, T[:, :, c].flatten(), rcond=None)
                    W[c] = w_c
                
                if it % 15 == 14:
                    M = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.max(np.abs(M - T)) < 1e-7:
                        break
            
            M = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.max(np.abs(M - T))
            
            if residual < 1e-5:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                M_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.max(np.abs(M_r - T)) < 1e-9:
                    found = (U_r, V_r, W_r)
                    break
        
        if found is not None:
            best_result = found
            best_rank = target_rank
            if target_rank <= 25:
                break
    
    U, V, W = best_result
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
