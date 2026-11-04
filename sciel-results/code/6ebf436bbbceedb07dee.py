import numpy as np
import time

def construct():
    start_time = time.time()
    max_time = 140
    
    n, m, p = 2, 3, 4
    target_rank = 19
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    T = np.ones((n, m, p))
    
    def verify_decomposition(U, V, W, R):
        """Verify that U, V, W with rank R exactly decompose the tensor."""
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(A_approx, T, atol=1e-9)
    
    def try_rank(R, max_restarts=50):
        """Try to find rank R decomposition for 2x3x4."""
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
        
        num_params = R * (n * m + m * p + p * n)
        
        # Try different initialization strategies
        init_strategies = [
            'uniform',  # Random uniform
            'structured',  # Structured based on schoolbook
            'sparse'  # Sparse random
        ]
        
        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                return None
            
            strategy = init_strategies[attempt % len(init_strategies)]
            
            if strategy == 'uniform':
                init_params = np.random.uniform(-1, 1, size=num_params)
            elif strategy == 'structured':
                # Initialize based on schoolbook pattern with noise
                init_params = np.zeros(num_params)
                # Add small random perturbation
                init_params += np.random.normal(0, 0.1, size=num_params)
            else:  # sparse
                init_params = np.random.uniform(-1, 1, size=num_params)
                # Zero out 70% of entries
                mask = np.random.random(num_params) < 0.3
                init_params *= mask
            
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residuals, init_params, method='trf', max_nfev=300)
                
                if sol.cost < 1e-4:
                    # Try to snap to integers or half-integers
                    # First try rounding to integers
                    rounded_int = np.round(sol.x)
                    res_int = residuals(rounded_int)
                    if np.linalg.norm(res_int) < 1e-8:
                        U, V, W = unpack(rounded_int)
                        if verify_decomposition(U, V, W, R):
                            return {"U": U, "V": V, "W": W}
                    
                    # Try rounding to half-integers (multiply by 2, round, divide by 2)
                    half_rounded = np.round(sol.x * 2) / 2
                    res_half = residuals(half_rounded)
                    if np.linalg.norm(res_half) < 1e-8:
                        U, V, W = unpack(half_rounded)
                        if verify_decomposition(U, V, W, R):
                            return {"U": U, "V": V, "W": W}
            except Exception:
                pass
        
        return None
    
    # Try to find rank 19 decomposition
    dec_234 = try_rank(19)
    
    if dec_234 and verify_decomposition(dec_234["U"], dec_234["V"], dec_234["W"], 19):
        return {
            "U": dec_234["U"].tolist(),
            "V": dec_234["V"].tolist(),
            "W": dec_234["W"].tolist()
        }
    
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
