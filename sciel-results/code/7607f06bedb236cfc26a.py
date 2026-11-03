import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p
    # Schoolbook
    U0 = np.zeros((n * m, rank_seed))
    V0 = np.zeros((m * p, rank_seed))
    W0 = np.zeros((p * n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_seed
    best_residual = 0.0

    # Target Tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Try to find lower rank
    # We will try ranks from 59 down to 50
    # Time budget: 120s
    start_time = time.time()
    time_limit = 110.0 # Safety margin

    for R in range(rank_seed - 1, max(10, rank_seed - 20), -1):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random values or modified schoolbook
        # Using random init for exploration
        np.random.seed(R) # Reproducibility for debugging, but maybe random is better for search
        # Let's try a few random inits
        for init_iter in range(5):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_curr = x[:n*m*R].reshape(n*m, R)
                V_curr = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_curr = x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Compute tensor
                # T_est = sum_c u_c v_c w_c
                # Using einsum: 'ac, bc, dc -> abd'
                T_est = np.einsum('ac,bc,dc->abd', U_curr, V_curr, W_curr)
                residual = T - T_est
                return np.sum(residual**2)
            
            # Gradient
            def gradient(x):
                U_curr = x[:n*m*R].reshape(n*m, R)
                V_curr = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_curr = x[n*m*R + m*p*R :].reshape(p*n, R)
                
                T_est = np.einsum('ac,bc,dc->abd', U_curr, V_curr, W_curr)
                residual = T - T_est
                
                # dL/dU = -2 * residual * (V outer W)
                # residual shape (n*m, m*p, p*n)
                # V_curr shape (m*p, R)
                # W_curr shape (p*n, R)
                # grad_U[a, c] = -2 * sum_{b,d} residual[a,b,d] V_curr[b,c] W_curr[d,c]
                # einsum: 'abd, bc, dc -> ac'
                grad_U = -2 * np.einsum('abd,bc,dc->ac', residual, V_curr, W_curr)
                grad_V = -2 * np.einsum('abd,ac,dc->bc', residual, U_curr, W_curr)
                grad_W = -2 * np.einsum('abd,ac,bc->dc', residual, U_curr, V_curr)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

            # Optimize
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 50})
                
                if res.fun < 1e-4: # Found a solution?
                    # Check if it's exact
                    x_opt = res.x
                    U_opt = x_opt[:n*m*R].reshape(n*m, R)
                    V_opt = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    T_est = np.einsum('ac,bc,dc->abd', U_opt, V_opt, W_opt)
                    if np.allclose(T_est, T, atol=1e-6):
                        best_rank = R
                        best_U = U_opt
                        best_V = V_opt
                        best_W = W_opt
                        # Try to round to half-integers
                        # If successful, return immediately
                        # For now, just record it
                        print(f"Found rank {R} with residual {res.fun}")
                        
            except Exception as e:
                pass

    # If a lower rank was found, we need to format the output
    # The problem asks for integer/half-integer entries.
    # If we found a floating point solution, we should try to round it.
    # If rounding fails, we might still return the floating point solution?
    # "exactly decompose" implies we need exactness.
    # If we can't round, we should probably return the schoolbook solution to be safe.
    # However, if the residual is extremely small, it might be a valid rank R tensor over reals.
    # But the contract says "integer/half-integer entries".
    
    # Let's try to round the found solution
    if best_rank < rank_seed:
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        T_rounded = np.einsum('ac,bc,dc->abd', U_rounded, V_rounded, W_rounded)
        if np.allclose(T_rounded, T, atol=1e-6):
            # Convert to lists
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
    
    # Fallback to schoolbook
    return {
        "U": U0.tolist(),
        "V": V0.tolist(),
        "W": W0.tolist()
    }
