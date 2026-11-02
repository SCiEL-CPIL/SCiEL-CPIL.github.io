import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [5, 5, 5]
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 1. Initialize with Schoolbook Decomposition (Rank 125)
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
                
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank

    # Helper: Compute Tensor T from U, V, W
    def compute_tensor(U_mat, V_mat, W_mat):
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Efficiently using einsum
        # U shape: (n*m, R) -> reshape to (n, m, R)
        # V shape: (m*p, R) -> reshape to (m, p, R)
        # W shape: (p*n, R) -> reshape to (p, n, R)
        
        U_reshaped = U_mat.reshape(n, m, -1)
        V_reshaped = V_mat.reshape(m, p, -1)
        W_reshaped = W_mat.reshape(p, n, -1)
        
        # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        # Note: W is indexed by k*n+i, so W[k, i, c]
        
        # Use einsum to compute the sum over c
        # 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return T

    # Helper: Error function for optimization
    # We flatten U, V, W into a single vector x
    def error_func(x):
        U_flat = x[:n*m*best_rank]
        V_flat = x[n*m*best_rank : n*m*best_rank + m*p*best_rank]
        W_flat = x[n*m*best_rank + m*p*best_rank : n*m*best_rank + m*p*best_rank + p*n*best_rank]
        
        U_curr = U_flat.reshape(n*m, best_rank)
        V_curr = V_flat.reshape(m*p, best_rank)
        W_curr = W_flat.reshape(p*n, best_rank)
        
        T = compute_tensor(U_curr, V_curr, W_curr)
        # Target is all ones
        error = np.sum((T - 1.0)**2)
        return error

    # Helper: Check if a decomposition is valid
    def is_valid(U_mat, V_mat, W_mat, r):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            return False
        T = compute_tensor(U_mat, V_mat, W_mat)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-8)

    # Strategy: Try to reduce rank by setting columns to zero and re-optimizing.
    # We will iterate through ranks from current best down to a lower bound (e.g., 40).
    
    # To save time, we only try a few random column removals per rank.
    # We start from the schoolbook and try to "merge" or "remove" columns.
    
    # A more effective search for small N:
    # Try random initializations for lower ranks and optimize.
    
    # However, random init for rank 49 is very hard to converge.
    # Let's try to improve the schoolbook by local search.
    
    # Local Search: 
    # 1. Take current best.
    # 2. Pick a column c.
    # 3. Set it to zero.
    # 4. Optimize the remaining columns to minimize error.
    # 5. If error is ~0, accept.
    
    # We'll try this for a few iterations.
    
    current_U = best_U
    current_V = best_V
    current_W = best_W
    current_rank = best_rank
    
    # We'll try to reduce rank by 1 at a time, with a limited number of attempts.
    # Given the 120s limit, we can do maybe 10-20 heavy optimization steps.
    
    max_rank_to_try = 120 # Start close to 125
    min_rank_to_try = 45 # Known lower bound is around 49, so 45 is a safe lower bound for search
    
    # We will iterate from current_rank down to min_rank_to_try
    # But doing this sequentially is slow.
    # Instead, we will do a random restart search for specific target ranks.
    
    # Let's try to find a rank 49 decomposition directly using a known structure if possible, 
    # or use a fast local search.
    
    # For now, let's implement a simple local search that tries to zero out columns.
    
    attempts_per_rank = 5
    for target_rank in range(current_rank - 1, min_rank_to_try - 1, -1):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Try to find a decomposition of target_rank
        # Initialize with the best_U, best_V, best_W truncated? 
        # No, better to start from a random sparse initialization or the schoolbook with zeros.
        
        # We'll use the current best as a starting point, but we need to reduce rank.
        # If current_rank > target_rank, we can't just truncate.
        
        # Let's try a different approach: Random Initialization + Optimization
        # This is expensive but might find a solution.
        
        # Given the time limit, let's just stick to the schoolbook if we can't find a better one quickly.
        # But we can try to optimize the schoolbook to see if we can make it sparse?
        # Schoolbook is already sparse (one 1 per column).
        
        # Let's try to verify if the schoolbook is indeed the best we can find quickly.
        # The score is 1.0 (rank 125).
        
        # If we find a valid decomposition with rank < 125, we return it.
        
        # To increase the chance, let's try a few random rank-50 initializations.
        # If we find one, it's a huge win.
        
        # We'll spend 50% of the time trying random rank-50 searches.
        
        time_spent = time.time() - start_time
        time_budget_for_search = TIME_LIMIT - SAFETY_MARGIN - time_spent
        if time_budget_for_search < 5.0:
            break
            
        # Try a few random rank 50 decompositions
        num_random_trials = 3
        for trial in range(num_random_trials):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            r = 50 # Target rank
            # Random initialization
            U_rand = np.random.randn(n * m, r)
            V_rand = np.random.randn(m * p, r)
            W_rand = np.random.randn(p * n, r)
            
            # Scale to help convergence
            U_rand *= 0.1
            V_rand *= 0.1
            W_rand *= 0.1
            
            x0 = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
            
            # Optimize
            try:
                # Use Nelder-Mead or BFGS? BFGS is faster for smooth functions.
                # But the landscape is non-convex.
                res = minimize(error_func, x0, method='BFGS', options={'maxiter': 100, 'disp': False})
                
                if res.fun < 1e-8:
                    U_sol = res.x[:n*m*r].reshape(n*m, r)
                    V_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_sol = res.x[n*m*r + m*p*r : n*m*r + m*p*r + p*n*r].reshape(p*n, r)
                    
                    if is_valid(U_sol, V_sol, W_sol, r):
                        best_U = U_sol
                        best_V = V_sol
                        best_W = W_sol
                        best_rank = r
                        # We found a good one, we can stop or try to improve further
                        # Let's try to reduce rank further
                        break 
                        
            except Exception:
                continue
                
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
