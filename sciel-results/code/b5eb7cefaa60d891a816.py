"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares, minimize

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # safety margin before 150s kill

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

    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1))
        return max_err

    def make_residuals(r):
        """Create residual function for given rank r."""
        def residuals(x):
            U = x[:r*n*m].reshape(n*m, r)
            V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
            W = x[r*n*m + r*m*p:].reshape(p*n, r)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        res.append(val - 1)
            return res
        return residuals

    def make_residuals_smooth(r):
        """Create residual function that's smoother for optimization."""
        def residuals(x):
            U = x[:r*n*m].reshape(n*m, r)
            V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
            W = x[r*n*m + r*m*p:].reshape(p*n, r)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        res.append(val - 1)
            return res
        return residuals

    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But tensor rank can be lower. Let's try from 28 upward.
    # Actually, the minimum rank for matrix multiplication is at least max(n*m, m*p, p*n)
    # but that's not quite right. The rank must be at least the max of the three dimensions' products.
    # Actually no - the rank must be at least max(n*m, m*p, p*n) is not a valid lower bound.
    # A better lower bound: the tensor has n*m*p = 84 entries, and each rank-1 term gives 
    # at most... actually the lower bound is more subtle.
    # Let's just try from a reasonable range.
    
    for r in range(28, rank_school):
        if time.time() > deadline:
            break
        
        n_vars = r * (n*m + m*p + p*n)
        n_eqs = n * m * p
        
        # Need enough variables
        if n_vars < n_eqs:
            continue
        
        # Try multiple random initializations with least_squares
        found = False
        n_attempts = 15 if r <= 40 else 8
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Random normal
                x0 = np.random.randn(n_vars)
            elif attempt == 1:
                # Sparse initialization
                x0 = np.zeros(n_vars)
                n_nonzero = min(n_vars, 50)
                idx = np.random.choice(n_vars, n_nonzero, replace=False)
                x0[idx] = np.random.choice([-1, 1], n_nonzero)
            elif attempt == 2:
                # Small random
                x0 = np.random.randn(n_vars) * 0.1
            elif attempt == 3:
                # Structured: try to exploit symmetry
                x0 = np.random.randn(n_vars)
                # Scale down W
                w_start = r*n*m + r*m*p
                x0[w_start:] *= 0.5
            else:
                x0 = np.random.randn(n_vars) * (0.5 + 0.5 * attempt)
            
            try:
                residuals = make_residuals_smooth(r)
                result = least_squares(residuals, x0, method='lm', 
                                       max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                
                if result.success or result.cost < 1e-10:
                    sol = result.x
                    U_sol = sol[:r*n*m].reshape(n*m, r)
                    V_sol = sol[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                    W_sol = sol[r*n*m + r*m*p:].reshape(p*n, r)
                    
                    err = verify(U_sol, V_sol, W_sol)
                    if err < 1e-8:
                        # Polish with higher precision
                        try:
                            residuals_high = make_residuals_smooth(r)
                            result2 = least_squares(residuals_high, sol, method='lm',
                                                   max_nfev=10000, xtol=1e-15, ftol=1e-15, gtol=1e-15)
                            if result2.cost < 1e-12:
                                sol = result2.x
                                U_sol = sol[:r*n*m].reshape(n*m, r)
                                V_sol = sol[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                                W_sol = sol[r*n*m + r*m*p:].reshape(p*n, r)
                                err = verify(U_sol, V_sol, W_sol)
                        except:
                            pass
                        
                        if err < 1e-8:
                            best_rank = r
                            best_U, best_V, best_W = U_sol.copy(), V_sol.copy(), W_sol.copy()
                            found = True
                            break
            except Exception:
                pass
        
        if found:
            break
    
    # Try to find even lower ranks with more attempts if we have time
    if best_rank > 28 and time.time() < deadline - 20:
        for r in range(28, best_rank):
            if time.time() > deadline:
                break
            n_vars = r * (n*m + m*p + p*n)
            if n_vars < n * m * p:
                continue
            
            for attempt in range(20):
                if time.time() > deadline:
                    break
                x0 = np.random.randn(n_vars)
                try:
                    residuals = make_residuals_smooth(r)
                    result = least_squares(residuals, x0, method='lm',
                                           max_nfev=8000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                    if result.cost < 1e-10:
                        sol = result.x
                        U_sol = sol[:r*n*m].reshape(n*m, r)
                        V_sol = sol[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                        W_sol = sol[r*n*m + r*m*p:].reshape(p*n, r)
                        err = verify(U_sol, V_sol, W_sol)
                        if err < 1e-8:
                            best_rank = r
                            best_U, best_V, best_W = U_sol.copy(), V_sol.copy(), W_sol.copy()
                            break
                except:
                    pass
            if best_rank == r:
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
                val = np.dot(U_out[i*m+j, :], V_out[j*p+k, :] * W_out[k*n+i, :])
                max_err = max(max_err, abs(val - 1))

    if max_err > 1e-6:
        # Try rounding to integers
        U_out_int = np.round(best_U)
        V_out_int = np.round(best_V)
        W_out_int = np.round(best_W)
        max_err_int = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_out_int[i*m+j, :], V_out_int[j*p+k, :] * W_out_int[k*n+i, :])
                    max_err_int = max(max_err_int, abs(val - 1))
        
        if max_err_int < 1e-6:
            U_out, V_out, W_out = U_out_int, V_out_int, W_out_int
        else:
            # Use the unrounded solution
            U_out, V_out, W_out = best_U, best_V, best_W

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
