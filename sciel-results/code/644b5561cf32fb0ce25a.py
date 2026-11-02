import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    n_sub, m_sub, p_sub = 2, 3, 2
    repeats = p // p_sub
    
    deadline = time.time() + 115
    
    best_rank_sub = None
    best_U_sub = None
    best_V_sub = None
    best_W_sub = None
    
    lower_bound_sub = max(n_sub * m_sub, m_sub * p_sub, p_sub * n_sub)
    
    for r_sub in range(lower_bound_sub, lower_bound_sub + 6):
        if time.time() > deadline:
            break
            
        success, U_sub, V_sub, W_sub = _find_decomposition_sub(n_sub, m_sub, p_sub, r_sub, deadline)
        if success:
            best_rank_sub = r_sub
            best_U_sub = U_sub
            best_V_sub = V_sub
            best_W_sub = W_sub
            break
            
    if best_rank_sub is not None:
        full_rank = best_rank_sub * repeats
        U_full = [[0] * full_rank for _ in range(n * m)]
        V_full = [[0] * full_rank for _ in range(m * p)]
        W_full = [[0] * full_rank for _ in range(p * n)]
        
        for rep in range(repeats):
            offset = rep * best_rank_sub
            
            for i in range(n_sub):
                for j in range(m_sub):
                    u_idx_sub = i * m_sub + j
                    u_idx_full = i * m + j
                    for c in range(best_rank_sub):
                        U_full[u_idx_full][offset + c] = best_U_sub[u_idx_sub][c]
                        
            for j in range(m_sub):
                for k_sub in range(p_sub):
                    k_full = rep * p_sub + k_sub
                    v_idx_sub = j * p_sub + k_sub
                    v_idx_full = j * p + k_full
                    for c in range(best_rank_sub):
                        V_full[v_idx_full][offset + c] = best_V_sub[v_idx_sub][c]
                        
            for k_sub in range(p_sub):
                k_full = rep * p_sub + k_sub
                for i in range(n_sub):
                    w_idx_sub = k_sub * n_sub + i
                    w_idx_full = k_full * n + i
                    for c in range(best_rank_sub):
                        W_full[w_idx_full][offset + c] = best_W_sub[w_idx_sub][c]
                        
        return {"U": U_full, "V": V_full, "W": W_full}
        
    # Fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_decomposition_sub(n, m, p, r, deadline):
    random.seed(42)
    
    while time.time() < deadline:
        U = np.array([[random.randint(-1, 1) for _ in range(r)] for _ in range(n*m)], dtype=float)
        W = np.array([[random.randint(-1, 1) for _ in range(r)] for _ in range(p*n)], dtype=float)
        
        V_sol = np.zeros((m*p, r), dtype=float)
        valid = True
        
        for j in range(m):
            for k in range(p):
                A_sys = np.zeros((n, r), dtype=float)
                for i in range(n):
                    for c in range(r):
                        A_sys[i, c] = U[i*m+j, c] * W[k*n+i, c]
                
                if np.linalg.matrix_rank(A_sys) < n:
                    valid = False
                    break
                    
                V_col, _, _, _ = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)
                
                if not np.allclose(A_sys @ V_col, np.ones(n), atol=1e-6):
                    valid = False
                    break
                    
                V_col_2 = V_col * 2
                if not np.allclose(V_col_2, np.round(V_col_2), atol=1e-6):
                    valid = False
                    break
                    
                V_sol[j*p+k, :] = V_col
                
        if not valid:
            continue
            
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j, c] * V_sol[j*p+k, c] * W[k*n+i, c]
                    if not np.isclose(val, 1, atol=1e-6):
                        ok = False
                        break
                if not ok: break
            if not ok: break
            
        if ok:
            return True, U.tolist(), V_sol.tolist(), W.tolist()
            
    return False, None, None, None
