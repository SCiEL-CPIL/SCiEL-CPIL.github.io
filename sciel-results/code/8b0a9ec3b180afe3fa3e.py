"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    max_time = 145  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    target_rank = n * m * p  # 24
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    T = np.ones((n, m, p))
    
    def verify_decomposition(U, V, W, R):
        """Verify if U, V, W form a valid decomposition of T."""
        U3 = np.array(U, dtype=float).reshape(n, m, R)
        V3 = np.array(V, dtype=float).reshape(m, p, R)
        W3 = np.array(W, dtype=float).reshape(p, n, R)
        A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(A_approx, T, atol=1e-9)
    
    def try_rank(R, max_restarts=30, max_nfev=500):
        """Try to find a rank-R decomposition using least squares optimization."""
        
        def unpack(params):
            U = params[:R * n * m].reshape(R, n * m).T
            V = params[R * n * m : R * (n * m + m * p)].reshape(R, m * p).T
            W = params[R * (n * m + m * p) :].reshape(R, p * n).T
            return U, V, W

        def residuals(params):
            U, V, W = unpack(params)
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            return (T - A_approx).flatten()
        
        def try_polish(params):
            """Try to polish a solution to exact integer/half-integer values."""
            # Round to nearest 0.5
            rounded = np.round(params * 2) / 2.0
            res = residuals(rounded)
            if np.linalg.norm(res) < 1e-6:
                U, V, W = unpack(rounded)
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": R}
            
            # Try rounding to nearest integer
            rounded_int = np.round(params)
            res_int = residuals(rounded_int)
            if np.linalg.norm(res_int) < 1e-6:
                U, V, W = unpack(rounded_int)
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": R}
            
            return None

        num_params = R * (n * m + m * p + p * n)
        
        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                return None
            
            # Random initialization with small integers
            # Use different strategies for different attempts
            if attempt % 3 == 0:
                init_params = np.random.choice([-1, 0, 1], size=num_params)
            elif attempt % 3 == 1:
                init_params = np.random.choice([-0.5, 0, 0.5], size=num_params)
            else:
                init_params = np.random.randn(num_params) * 0.5
            
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residuals, init_params, method='trf', max_nfev=max_nfev)
                
                # Check if we found a good solution
                if sol.cost < 1e-10:
                    result = try_polish(sol.x)
                    if result:
                        return result
                elif sol.cost < 1e-6:
                    # Try polishing even if not perfect
                    result = try_polish(sol.x)
                    if result:
                        return result
            except Exception:
                pass
                
        return None
    
    # Try lower ranks from 23 down to 15
    # Known lower bounds suggest rank around 20-21 for 2x3x4
    for R in range(target_rank - 1, 14, -1):
        if time.time() - start_time > max_time - 10:
            break
        
        res = try_rank(R)
        if res:
            return res
    
    # Fallback: Schoolbook decomposition
    rank = target_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
