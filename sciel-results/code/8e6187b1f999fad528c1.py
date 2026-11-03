import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def error_for_rank(R):
        """
        Attempts to find a decomposition of rank R.
        Returns (rank, U, V, W) if successful, else (None, None, None, None).
        """
        start_time = time.time()
        timeout = 10  # seconds per rank attempt to stay well within limits
        
        # Random initialization
        # Using small random values to avoid divergence
        np.random.seed(42) # For reproducibility in this block, though multiple restarts need different seeds
        
        best_solution = None
        best_error = np.inf

        # Try a few restarts
        for attempt in range(5):
            if time.time() - start_time > timeout:
                break
            
            # Initialize U, V, W
            U_init = np.random.randn(n*m, R) * 0.1
            V_init = np.random.randn(m*p, R) * 0.1
            W_init = np.random.randn(p*n, R) * 0.1
            
            # Flatten parameters
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Define objective function
            def objective(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Compute tensor product sum
                # T_comp[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                # Efficient computation:
                # Sum over r of U[:,r] @ V[:,r]^T gives intermediate? No.
                # We need elementwise product summed over r.
                # T_comp = np.einsum('ar,br,cr->abc', U, V, W)
                # This might be slow if done in python loops, but numpy handles it.
                
                T_comp = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_comp - T
                return np.sum(diff**2)

            # Run optimization
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.success or res.fun < 1e-5:
                # Check if solution is close to integers
                U_opt = res.x[:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Check integrality
                # If max deviation from nearest integer is small, snap to integers
                # We allow half-integers if needed? The prompt says "integer/half-integer".
                # Let's snap to nearest 0.5
                U_snap = np.round(U_opt * 2) / 2.0
                V_snap = np.round(V_opt * 2) / 2.0
                W_snap = np.round(W_opt * 2) / 2.0
                
                # Verify the snapped solution
                T_check = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                error_snap = np.sum((T_check - T)**2)
                
                if error_snap < 1e-9:
                    # Success
                    if res.fun < best_error:
                        best_error = res.fun
                        best_solution = (U_snap, V_snap, W_snap)

        if best_solution is not None:
            U, V, W = best_solution
            # Convert to lists of lists as required
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return R, U_list, V_list, W_list
        else:
            return None, None, None, None

    # Try to find a better rank than schoolbook
    # Start from rank_school - 1 down to rank_school - 5 (searching small space)
    # In reality, the drop is often small for non-cubic matrices, but let's try.
    for R in range(rank_school - 1, rank_school - 6, -1):
        if time.time() > 110: # Hard safety margin
            break
        result_rank, U_list, V_list, W_list = error_for_rank(R)
        if result_rank is not None:
            return {
                "U": U_list,
                "V": V_list,
                "W": W_list
            }

    # Fallback to schoolbook
    rank = rank_school
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
