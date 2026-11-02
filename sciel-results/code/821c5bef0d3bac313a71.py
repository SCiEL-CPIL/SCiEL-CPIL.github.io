"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
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
    
    # Precompute indices for faster evaluation
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def residuals(x, r):
        U = x[:r*n*m].reshape(n*m, r)
        V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
        W = x[r*n*m + r*m*p:].reshape(p*n, r)
        
        res = []
        for i, j, k in indices:
            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            res.append(val - 1)
        return res
    
    def try_rank(r):
        n_vars = r * (n*m + m*p + p*n)
        
        for init in range(5):
            if time.time() > deadline:
                return False
            
            x0 = np.random.randn(n_vars) * 0.5
            
            try:
                sol = least_squares(residuals, x0, args=(r,), method='lm', max_nfev=1000)
                
                # Check residual
                res_final = sol.fun
                max_err = np.max(np.abs(res_final))
                
                if max_err < 1e-8:
                    U_sol = sol.x[:r*n*m].reshape(n*m, r)
                    V_sol = sol.x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                    W_sol = sol.x[r*n*m + r*m*p:].reshape(p*n, r)
                    
                    # Try to snap to half-integers
                    U_half = np.round(U_sol * 2) / 2
                    V_half = np.round(V_sol * 2) / 2
                    W_half = np.round(W_sol * 2) / 2
                    
                    # Verify half-integer solution
                    max_err_half = 0
                    for i, j, k in indices:
                        val = np.dot(U_half[i*m+j, :], V_half[j*p+k, :] * W_half[k*n+i, :])
                        max_err_half = max(max_err_half, abs(val - 1))
                    
                    if max_err_half < 1e-8:
                        nonlocal best_rank, best_U, best_V, best_W
                        best_rank = r
                        best_U, best_V, best_W = U_half, V_half, W_half
                        return True
            except Exception:
                pass
        return False

    # Search for minimal rank
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    lower_bound = max(n*m, m*p, p*n)
    
    for r in range(lower_bound, rank_school):
        if time.time() > deadline:
            break
        
        if try_rank(r):
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
