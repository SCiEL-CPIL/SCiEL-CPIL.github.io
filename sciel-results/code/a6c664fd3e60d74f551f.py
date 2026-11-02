"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def residual_func(x, r):
        """Compute residual for given rank r."""
        U = x[:r*n*m].reshape(n*m, r)
        V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
        W = x[r*n*m + r*m*p:].reshape(p*n, r)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    res[idx] = np.dot(u_val, v_val * w_val) - 1
                    idx += 1
        return res
    
    # Try to find lower rank decompositions
    for r in range(80, rank_school):
        if time.time() > deadline:
            break
        
        n_vars = r * (n*m + m*p + p*n)
        
        found = False
        for init in range(10):
            if time.time() > deadline:
                break
            
            # Initialize from schoolbook, keeping first r columns, add noise
            if init == 0:
                U_init = U_school[:, :r].copy()
                V_init = V_school[:, :r].copy()
                W_init = W_school[:, :r].copy()
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                x0 = np.random.randn(n_vars)
            
            try:
                result = least_squares(
                    residual_func, 
                    x0, 
                    args=(r,), 
                    method='lm', 
                    max_nfev=500,
                    ftol=1e-8, 
                    xtol=1e-8, 
                    gtol=1e-8
                )
                
                if result.success or result.cost < 1e-10:
                    sol = result.x
                    U_sol = sol[:r*n*m].reshape(n*m, r)
                    V_sol = sol[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                    W_sol = sol[r*n*m + r*m*p:].reshape(p*n, r)
                    
                    # Verify exactness
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_sol[i*m+j, :] * V_sol[j*p+k, :] * W_sol[k*n+i, :])
                                max_err = max(max_err, abs(val - 1))
                    
                    if max_err < 1e-6:
                        best_rank = r
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    # Try to round to half-integers
    def try_round(arr, scale=2):
        rounded = np.round(arr * scale) / scale
        return rounded
    
    U_out = try_round(best_U)
    V_out = try_round(best_V)
    W_out = try_round(best_W)
    
    # Verify rounded
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_out[i*m+j, :] * V_out[j*p+k, :] * W_out[k*n+i, :])
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-6:
        U_out, V_out, W_out = best_U, best_V, best_W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
