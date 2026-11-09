import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Schoolbook solution as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = None
    best_rank = n * m * p

    # Reshape helper
    def reshape_factors(U, V, W, R):
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        return U3, V3, W3

    def compute_error(params, R):
        U, V, W = params
        U3, V3, W3 = reshape_factors(U, V, W, R)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        error = T - target
        return error.flatten()

    def verify_solution(U, V, W, R):
        U3, V3, W3 = reshape_factors(U, V, W, R)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    # Try ranks from a reasonable lower bound up to schoolbook rank
    # Lower bound for 3x3x8 is likely around 30-40 based on known results for similar sizes
    # We'll start from 30 to balance search time and potential improvement
    for R in range(30, best_rank + 1):
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts: more for lower ranks (harder), but time-constrained
        # Let's do 10 restarts for R < 40, 5 for R < 60, 3 otherwise
        if R < 40:
            num_restarts = 10
        elif R < 60:
            num_restarts = 5
        else:
            num_restarts = 3

        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization with scaling
            # Scale by 1/R^(1/3) so that sum of R products is O(1)
            scale = 1.0 / (R ** (1.0 / 3.0))
            U = np.random.randn(n * m, R) * scale
            V = np.random.randn(m * p, R) * scale
            W = np.random.randn(p * n, R) * scale
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use 'trf' method which can handle bounds if needed, but 'lm' is faster for unconstrained
                # However, 'lm' requires m >= n (number of residuals >= number of variables)
                # Residuals = n*m*p = 72
                # Variables = (n*m + m*p + p*n) * R = (9 + 24 + 24) * R = 57 * R
                # For R > 1, variables > residuals, so 'lm' might fail or be inefficient.
                # Let's stick to 'lm' but be aware. Actually, 'lm' in scipy can handle underdetermined?
                # Docs: "method='lm' : Levenberg-Marquardt. Does not support bounds or complex numbers.
                #        The number of variables must be no greater than the number of the function's elements, f."
                # So for R >= 2, 57*R > 72 is false for R=1 (57<=72), but for R=2, 114>72.
                # So 'lm' will fail for R >= 2.
                # We must use 'trf' or 'dogbox'. 'trf' is general purpose.
                res = least_squares(compute_error, params, args=(R,), method='trf', max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                # Found a solution
                U_sol = res.x[:n*m*R].reshape(n*m, R)
                V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify exactness
                if verify_solution(U_snap, V_snap, W_snap, R):
                    best_rank = R
                    best_result = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    found = True
                    break
        
        if found:
            break

    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
