"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import fsolve, least_squares

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
    
    def make_equations(r):
        """Create the system of equations for given rank r."""
        def equations(x):
            U = x[:r*n*m].reshape(n*m, r)
            V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
            W = x[r*n*m + r*m*p:].reshape(p*n, r)
            
            eqs = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        eqs.append(val - 1)
            return eqs
        return equations
    
    # Precompute indices for faster residual calculation
    # Residual r_ijk = sum_t U[i,j,t] V[j,k,t] W[k,i,t] - 1
    # We want to minimize sum (r_ijk)^2
    
    def residual_func(x, r):
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
    # We search from low rank upwards. If we find a solution, we stop.
    # Start from a reasonable lower bound. 
    # Lower bound for M_{n,m,p} is at least max(n*m, m*p, p*n) ? No.
    # For 3x4x7, schoolbook is 84.
    # Let's try ranks from 80 to 83.
    
    for r in range(80, rank_school):
        if time.time() > deadline:
            break
        
        n_vars = r * (n*m + m*p + p*n)
        
        # Try multiple random initializations with least_squares
        found = False
        for init in range(20):
            if time.time() > deadline:
                break
            
            x0 = np.random.randn(n_vars)
            
            try:
                # Use least_squares with 'lm' or 'trf'
                # 'lm' is Levenberg-Marquardt, good for small/medium problems
                # 'trf' is Trust Region Reflective, good for bounded or large
                # We don't have bounds, so 'lm' or 'trf' (unbounded)
                # 'lm' is often faster for these dense problems if not too large.
                result = least_squares(
                    residual_func, 
                    x0, 
                    args=(r,), 
                    method='lm', 
                    max_nfev=2000,
                    ftol=1e-10, 
                    xtol=1e-10, 
                    gtol=1e-10
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
                    
                    if max_err < 1e-8:
                        best_rank = r
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    # Convert to half-integers if possible
    def to_half_integer(arr):
        return np.round(arr * 2) / 2
    
    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)
    
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
