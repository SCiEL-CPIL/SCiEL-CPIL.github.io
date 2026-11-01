import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find lower rank
    # We iterate from rank_schoolbook - 1 downwards
    # We limit the search to a few ranks to ensure we finish in time
    max_search_rank = rank_schoolbook - 1
    min_search_rank = max(30, rank_schoolbook - 10) # Search down to 50 or so
    
    for r in range(max_search_rank, min_search_rank - 1, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random values
            # Scaling is important for convergence
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Flatten for optimization
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_f = x[:n*m*r].reshape(n*m, r)
                V_f = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_f = x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Compute error
                # Error[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1
                # We can vectorize this
                # Create indices
                # This is a 3-way tensor contraction
                # Sum over c
                # We can use np.einsum or manual loops for small sizes
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U_f[i * m + j]
                            v_val = V_f[j * p + k]
                            w_val = W_f[k * n + i]
                            s = np.dot(u_val, v_val) * w_val # No, this is not right.
                            # It is sum_c u_c v_c w_c
                            s = np.sum(u_val * v_val * w_val)
                            err += (s - 1.0) ** 2
                return err

            # Use a simple gradient descent or L-BFGS-B
            # L-BFGS-B might be slow if called many times.
            # Let's try a few steps of gradient descent with numerical gradient?
            # Or use scipy.optimize.minimize with method='BFGS'
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-6})
                if res.success and res.fun < 1e-8:
                    # Extract solution
                    U_opt = res.x[:n*m*r].reshape(n*m, r)
                    V_opt = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_opt = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    # Check if close to half-integers
                    # Multiply by 2 and check closeness to integers
                    U_2 = U_opt * 2
                    V_2 = V_opt * 2
                    W_2 = W_opt * 2
                    
                    if np.allclose(U_2, np.round(U_2), atol=1e-5) and \
                       np.allclose(V_2, np.round(V_2), atol=1e-5) and \
                       np.allclose(W_2, np.round(W_2), atol=1e-5):
                       
                       U_int = np.round(U_2).astype(int) / 2.0
                       V_int = np.round(V_2).astype(int) / 2.0
                       W_int = np.round(W_2).astype(int) / 2.0
                       
                       # Verify exactness
                       valid = True
                       for i in range(n):
                           for j in range(m):
                               for k in range(p):
                                   s = np.sum(U_int[i * m + j] * V_int[j * p + k] * W_int[k * n + i])
                                   if not np.isclose(s, 1.0, atol=1e-9):
                                       valid = False
                                       break
                               if not valid: break
                           if not valid: break
                       if not valid: break
                       
                       if valid:
                           best_rank = r
                           best_result = {
                               "U": U_int.tolist(),
                               "V": V_int.tolist(),
                               "W": W_int.tolist()
                           }
                           # We can break early if we find a good one, but maybe lower is better
                           # Continue searching for lower ranks?
                           # Given time, break out of attempt loop
                           break
            except Exception:
                pass

    return best_result
