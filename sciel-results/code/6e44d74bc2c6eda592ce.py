import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Define the tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Try different ranks, starting from a lower bound
    # The minimum rank is at least max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But for matrix multiplication, it's likely higher
    
    # Start with schoolbook rank
    rank_school = n * m * p
    
    # Schoolbook solution
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # We'll try ranks from 20 up to 60
    
    best_solution = (U_school, V_school, W_school)
    best_rank = rank_school
    
    for rank in range(20, rank_school + 1):
        if time.time() > deadline - 30:  # Leave time for final check
            break
        
        # Try to find a rank-r decomposition
        # Initialize with random vectors
        np.random.seed(42)  # For reproducibility
        U_init = np.random.randn(n * m, rank)
        V_init = np.random.randn(m * p, rank)
        W_init = np.random.randn(p * n, rank)
        
        # Define the objective function
        def objective(params):
            U = params[:n*m*rank].reshape(n * m, rank)
            V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m * p, rank)
            W = params[n*m*rank + m*p*rank:].reshape(p * n, rank)
            
            # Compute the reconstructed tensor
            T_recon = np.zeros((n * m, m * p, p * n))
            for r in range(rank):
                T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            # Compute the error
            error = np.sum((T - T_recon) ** 2)
            return error
        
        # Try optimization
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(objective, params_init, method='BFGS', 
                            options={'maxiter': 1000, 'gtol': 1e-10})
            
            if result.success and result.fun < 1e-8:
                U_opt = result.x[:n*m*rank].reshape(n * m, rank)
                V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m * p, rank)
                W_opt = result.x[n*m*rank + m*p*rank:].reshape(p * n, rank)
                
                # Verify the solution
                T_recon = np.zeros((n * m, m * p, p * n))
                for r in range(rank):
                    T_recon += np.einsum('i,j,k->ijk', U_opt[:, r], V_opt[:, r], W_opt[:, r])
                
                if np.allclose(T_recon, T, atol=1e-8):
                    best_solution = (U_opt, V_opt, W_opt)
                    best_rank = rank
                    break  # Found the minimum rank, stop searching
        except Exception as e:
            continue
    
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
