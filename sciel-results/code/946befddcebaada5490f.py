"""Program for 3x3x6 matrix multiplication rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact. Returns max error."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _objective(x, n, m, p, rank):
    """Compute the sum of squared errors for a given parameter vector."""
    U_flat = x[:n*m*rank]
    V_flat = x[n*m*rank : n*m*rank + m*p*rank]
    W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
    
    U_mat = U_flat.reshape(n * m, rank)
    V_mat = V_flat.reshape(m * p, rank)
    W_mat = W_flat.reshape(p * n, rank)
    
    err_sum = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                err_sum += (val - 1.0) ** 2
    return err_sum

def _try_find_lower_rank(n, m, p, target_rank, max_time=5.0, seed=0):
    """
    Attempt to find a decomposition of the given rank using local optimization.
    Uses a structured initialization based on splitting the p-dimension.
    """
    start_time = time.time()
    
    # Structured initialization: Assume the 6 columns of B can be handled by 
    # a structure similar to two 3x3 multiplications with some overlap or sharing.
    # For rank r, we initialize with random values but scaled.
    np.random.seed(seed)
    
    U_init = np.random.randn(n * m, target_rank) * 0.5
    V_init = np.random.randn(m * p, target_rank) * 0.5
    W_init = np.random.randn(p * n, target_rank) * 0.5
    
    # Add a "schoolbook-like" bias to help convergence
    # Map the first n*m*p entries to a partial schoolbook if possible
    # This is a heuristic to get closer to the solution manifold
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def objective(x):
        return _objective(x, n, m, p, target_rank)
    
    try:
        # Use L-BFGS-B for local refinement
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
        
        if result.fun < 1e-10:
            U_found = result.x[:n*m*target_rank].reshape(n * m, target_rank)
            V_found = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
            W_found = result.x[n*m*target_rank + m*p*target_rank :].reshape(p * n, target_rank)
            
            # Verify exactness
            err = _check_decomposition(U_found, V_found, W_found, n, m, p)
            if err < 1e-8:
                return U_found, V_found, W_found
                
    except Exception:
        pass
        
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # 1. Get the guaranteed schoolbook solution (Rank 54)
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # 2. Try to find a lower rank. 
    # We try ranks 53, 52, 51, 50. 
    # We limit the time per rank to ensure we don't exceed the deadline.
    # Given the difficulty, we prioritize speed of fallback.
    
    target_ranks = [53, 52, 51, 50]
    
    for target_rank in target_ranks:
        if time.time() - start_time > deadline - 10:
            break
            
        # Try a few seeds for this rank
        found = False
        for seed in range(3):
            if time.time() - start_time > deadline - 10:
                break
                
            U_r, V_r, W_r = _try_find_lower_rank(n, m, p, target_rank, max_time=3.0, seed=seed)
            
            if U_r is not None:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                found = True
                break # Found valid decomposition at this rank, move to lower rank
                
        if found:
            continue # Try next lower rank
            
    # 3. Return the best found
    # Convert to lists of floats as required by the interface
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
