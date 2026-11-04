import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    max_time = 115  # Safety margin

    n, m, p = 2, 3, 4
    target_rank = 20
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))

    def unpack(params, R):
        U = params[:R * n * m].reshape(R, n * m).T
        V = params[R * n * m : R * (n * m + m * p)].reshape(R, m * p).T
        W = params[R * (n * m + m * p) :].reshape(R, p * n).T
        return U, V, W

    def residuals(params, R):
        U, V, W = unpack(params, R)
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return (T - A_approx).flatten()

    def try_rank(R, max_restarts=100):
        num_params = R * (n * m + m * p + p * n)
        
        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                return None
            
            # Random initialization
            # Use small integers/half-integers to help convergence to the discrete solution
            init_params = np.random.choice([-1, -0.5, 0, 0.5, 1], size=num_params)
            
            try:
                from scipy.optimize import least_squares
                sol = least_squares(
                    lambda x: residuals(x, R), 
                    init_params, 
                    method='trf', 
                    max_nfev=100
                )
                
                if sol.cost < 1e-8:
                    # Snap to nearest 0.5
                    rounded_params = np.round(sol.x * 2) / 2.0
                    res_check = residuals(rounded_params, R)
                    if np.linalg.norm(res_check) < 1e-9:
                        U, V, W = unpack(rounded_params, R)
                        # Final verification
                        U3 = U.reshape(n, m, R)
                        V3 = V.reshape(m, p, R)
                        W3 = W.reshape(p, n, R)
                        A_exact = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                        if np.allclose(A_exact, T, atol=1e-9):
                            return {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
            except Exception:
                pass
        return None

    # Attempt to find rank 20
    dec_20 = try_rank(20)
    
    if dec_20:
        return dec_20

    # Fallback to schoolbook rank 24
    rank = 24
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
