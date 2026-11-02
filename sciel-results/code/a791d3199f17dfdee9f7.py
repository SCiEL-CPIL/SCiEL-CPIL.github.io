import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank_candidates = [53, 52, 51]
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_solution = schoolbook()
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    for r in target_rank_candidates:
        if time.time() > deadline - 10:
            break
            
        # Try multiple random restarts
        # 53 is easier, give it more chances
        num_restarts = 20 if r == 53 else (15 if r == 52 else 10)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random values
            # Using uniform distribution might be better to avoid zero cancellations
            U = np.random.uniform(-1, 1, (n * m, r))
            V = np.random.uniform(-1, 1, (m * p, r))
            W = np.random.uniform(-1, 1, (p * n, r))
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals(x):
                U_flat = x[:n*m*r]
                V_flat = x[n*m*r : n*m*r + m*p*r]
                W_flat = x[n*m*r + m*p*r :]
                
                U_mat = U_flat.reshape(n*m, r)
                V_mat = V_flat.reshape(m*p, r)
                W_mat = W_flat.reshape(p*n, r)
                
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            try:
                sol = least_squares(residuals, x0, method='trf', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                error = np.sum(sol.fun**2)
                
                if error < 1e-6:
                    U_opt = sol.x[:n*m*r].reshape(n*m, r)
                    V_opt = sol.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_opt = sol.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    U_snap = np.round(U_opt * 2) / 2.0
                    V_snap = np.round(V_opt * 2) / 2.0
                    W_snap = np.round(W_opt * 2) / 2.0
                    
                    # Verify exactness
                    is_valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_snap[i*m+j, :] * V_snap[j*p+k, :] * W_snap[k*n+i, :])
                                if not np.isclose(val, 1.0, atol=1e-9):
                                    is_valid = False
                                    break
                            if not is_valid: break
                        if not is_valid: break
                    if not is_valid: break
                    
                    if is_valid:
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
                        
            except Exception:
                pass

    return best_solution
# EVOLVE-BLOCK-END
