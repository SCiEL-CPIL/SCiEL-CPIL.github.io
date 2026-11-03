import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    def check_half_integer(U, V, W, tol=1e-6):
        for mat in [U, V, W]:
            for val in mat:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
        return True
    
    def residual_func(params, r):
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        res = np.zeros((n, m, p))
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
        return (1 - res).flatten()
    
    # Try ranks from lower bound up
    min_rank = max(n*m, m*p, p*n)  # 21
    best_result = None
    best_rank = rank_school
    
    for r in range(min_rank, rank_school + 1):
        if time.time() > deadline:
            break
        if r >= best_rank:
            break
        
        # Try a few random restarts
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            rng = np.random.default_rng(restart * 1000 + r)
            U0 = rng.normal(0, 0.1, (n*m, r))
            V0 = rng.normal(0, 0.1, (m*p, r))
            W0 = rng.normal(0, 0.1, (p*n, r))
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = least_squares(residual_func, params0, args=(r,), method='lm', max_nfev=1000)
                if result.success or result.cost < 1e-8:
                    U = result.x[:n*m*r].reshape(n*m, r)
                    V = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify
                    if check_half_integer(U_snap, V_snap, W_snap):
                        # Check that the snapped solution is exact
                        res = np.zeros((n, m, p))
                        for c in range(r):
                            uc = U_snap[:, c]
                            vc = V_snap[:, c]
                            wc = W_snap[:, c]
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        res[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
                        if np.allclose(res, 1, atol=1e-9):
                            best_result = (U_snap, V_snap, W_snap)
                            best_rank = r
                            break
            except Exception:
                pass
        
        if best_result is not None and best_rank == r:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    else:
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
