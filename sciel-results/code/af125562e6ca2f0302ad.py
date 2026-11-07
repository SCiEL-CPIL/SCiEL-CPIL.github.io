"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Computes the Knowledge Gradient (KG) acquisition scores using Monte Carlo approximation.
    
    Parameters:
    mu: array of shape (N,), posterior means.
    sigma: array of shape (N,), posterior standard deviations.
    f_best: float, the best observed value so far.
    
    Returns:
    scores: array of shape (N,), KG scores (higher is better).
    """
    N = len(mu)
    # Number of Monte Carlo samples. 
    # 200 samples provides a good balance between accuracy and speed for N=2000.
    n_samples = 200
    
    # Initialize scores to 0
    kg_scores = np.zeros(N)
    
    # Deadline guard: ensure we finish well before 330s. 
    # This loop is very fast, but we check just in case of extreme slowdowns.
    start_time = time.time()
    timeout_limit = 300.0  # Safety margin
    
    for _ in range(n_samples):
        # Check time
        if time.time() - start_time > timeout_limit:
            break
            
        # Sample from the posterior for all candidates
        # y ~ N(mu, sigma^2)
        y = mu + sigma * np.random.randn(N)
        
        # Find the index of the minimum and the second minimum in y
        # np.argmin gives the first occurrence of the minimum
        idx_min = np.argmin(y)
        
        # To find the second minimum, we can temporarily set the min to inf and find the min again
        # Or use a more efficient method. For N=2000, setting one value is fine.
        y_copy = y.copy()
        y_copy[idx_min] = np.inf
        idx_second_min = np.argmin(y_copy)
        
        # Construct the array of "min of all other candidates"
        # For all x != idx_min, min_{j!=x} y_j = y[idx_min]
        # For x == idx_min, min_{j!=x} y_j = y[idx_second_min]
        min_rest = np.full(N, y[idx_min])
        min_rest[idx_min] = y[idx_second_min]
        
        # The new global minimum if we evaluate at x is min(f_best, y[x], min_rest[x])
        # Note: min_rest[x] is the min of the *other* candidates.
        # The set of all candidates is {y[x]} U {other candidates}.
        # So the min of the set is min(y[x], min_rest[x]).
        # The overall min including the current best f_best is min(f_best, y[x], min_rest[x]).
        
        new_min = np.minimum(f_best, np.minimum(y, min_rest))
        
        # The reduction in the minimum is f_best - new_min
        reduction = f_best - new_min
        
        kg_scores += reduction
        
    # Average over samples
    kg_scores /= n_samples
    
    # Ensure scores are finite
    kg_scores = np.nan_to_num(kg_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return kg_scores
# EVOLVE-BLOCK-END
