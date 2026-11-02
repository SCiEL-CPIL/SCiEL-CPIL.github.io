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
        
        for init in range(8):
            if time.time() > deadline:
                return False
            
            # Different initialization strategies
            if init == 0:
                x0 = np.random.randn(n_vars) * 0.5
            elif init == 1:
                x0 = np.random.randn(n_vars) * 0.3
            elif init == 2:
                x0 = np.random.randn(n_vars) * 0.7
            elif init == 3:
                # Sparse initialization
                x0 = np.random.randn(n_vars)
                mask = np.random.rand(n_vars) < 0.3
                x0[~mask] = 0
                x0[mask] *= 0.5
            elif init == 4:
                # Block structure initialization
                x0 = np.zeros(n_vars)
                u_part = x0[:r*n*m].reshape(n*m, r)
                v_part = x0[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                w_part = x0[r*n*m + r*m*p:].reshape(p*n, r)
                for rr in range(r):
                    for i in range(n):
                        for j in range(m):
                            if (i + j + rr) % 3 == 0:
                                u_part[i*m+j, rr] = np.random.randn() * 0.5
                    for j in range(m):
                        for k in range(p):
                            if (j + k + rr) % 3 == 0:
                                v_part[j*p+k, rr] = np.random.randn() * 0.5
                    for k in range(p):
                        for i in range(n):
                            if (k + i + rr) % 3 == 0:
                                w_part[k*n+i, rr] = np.random.randn() * 0.5
                x0 = np.concatenate([u_part.flatten(), v_part.flatten(), w_part.flatten()])
            else:
                x0 = np.random.randn(n_vars) * (0.2 + 0.1 * init)
            
            try:
                sol = least_squares(residuals, x0, args=(r,), method='lm', max_nfev=1500)
                
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
    
    # Try ranks from lower bound up, but with time limits
    for r in range(lower_bound, rank_school):
        if time.time() > deadline - 20:
            break
        
        if try_rank(r):
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
