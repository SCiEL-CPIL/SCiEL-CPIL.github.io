"""Matrix multiplication [3,5,6] rank-minimal decomposition."""
import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Target tensor: M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        # Compute the product tensor
        # M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape for efficient computation
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # For each (i,j,k), compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a bit tricky. Let's do it directly.
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_reshaped[i, j, c] * V_reshaped[j, k, c] * W_reshaped[k, i, c]
                    result[i, j, k] = val
        return result
    
    def objective(params):
        """Objective function for optimization."""
        rank = len(params) // 3
        U = params[:rank * n * m].reshape(n * m, rank)
        V = params[rank * n * m:rank * n * m + rank * m * p].reshape(m * p, rank)
        W = params[rank * n * m + rank * m * p:].reshape(p * n, rank)
        
        M = check_decomposition(U, V, W)
        # We want M[i,j,k] = 1 for all valid (i,j,k)
        target = np.ones((n, m, p))
        error = np.sum((M - target) ** 2)
        return error
    
    # Try to find decompositions of decreasing rank
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # First, get schoolbook as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = (U_school, V_school, W_school, best_rank)
    
    # Try ranks from 60 down to 30
    for rank in range(60, 29, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + trial)
            U0 = np.random.randn(n * m, rank) * 0.5
            V0 = np.random.randn(m * p, rank) * 0.5
            W0 = np.random.randn(p * n, rank) * 0.5
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Use Nelder-Mead or BFGS
            try:
                result = minimize(objective, params0, method='Nelder-Mead', 
                                  options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                
                if result.fun < 1e-6:
                    U_opt = result.x[:rank * n * m].reshape(n * m, rank)
                    V_opt = result.x[rank * n * m:rank * n * m + rank * m * p].reshape(m * p, rank)
                    W_opt = result.x[rank * n * m + rank * m * p:].reshape(p * n, rank)
                    
                    # Verify
                    M = check_decomposition(U_opt, V_opt, W_opt)
                    if np.allclose(M, np.ones((n, m, p)), atol=1e-5):
                        best_result = (U_opt, V_opt, W_opt, rank)
                        best_rank = rank
                        break  # Found valid decomposition at this rank
            except Exception:
                continue
        
        if best_rank < 60:
            break  # Found something better
    
    U, V, W, _ = best_result
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
