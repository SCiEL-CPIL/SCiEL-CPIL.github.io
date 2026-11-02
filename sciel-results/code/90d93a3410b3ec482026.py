import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120 - 15  # 105s safety margin
    
    # Schoolbook baseline: rank 54
    R_school = n * m * p
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, r] = 1.0
                V_school[j * p + k, r] = 1.0
                W_school[k * n + i, r] = 1.0
                r += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = R_school
    
    # Try to find lower-rank decompositions using numerical optimization
    # Target ranks to try (from low to high)
    target_ranks = [18, 19, 20, 21, 22, 24, 27, 30, 36, 40, 45, 50]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        found_valid = False
        max_attempts = 3
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 25:
                break
            
            # Initialize with random values
            np.random.seed(42 + attempt)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Define the tensor and objective function
            # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
            
            def objective(params):
                U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Compute the decomposed tensor
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_flat[i*m+j], np.dot(V_flat[j*p+k], W_flat[k*n+i]))
                            error += (val - 1.0) ** 2
                
                return error
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B optimizer
            try:
                result = minimize(objective, params, method='L-BFGS-B', 
                                  options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.success or result.fun < 1e-8:
                    U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Verify the decomposition
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_opt[i*m+j], np.dot(V_opt[j*p+k], W_opt[k*n+i]))
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-6:
                        found_valid = True
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {
                                "U": U_opt.tolist(), 
                                "V": V_opt.tolist(), 
                                "W": W_opt.tolist()
                            }
                        break
                        
            except Exception as e:
                pass
        
        if found_valid and target_rank < best_rank:
            break
    
    return best_result
