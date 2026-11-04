import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    rank = U.shape[1]
    T = np.zeros((n, m, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        T[i, j, j2, k] += u_mat[i, j] * v_mat[j2, k] * w_mat[k, i]
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    expected = 1.0 if j == j2 else 0.0
                    if abs(T[i, j, j2, k] - expected) > tol:
                        return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # ... DP and Construction logic ...
    
    # Fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    
    # Try optimized
    try:
        U_opt, V_opt, W_opt, rank_opt = build_matrices(n, m, p)
        if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
            return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    except Exception:
        pass
        
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
