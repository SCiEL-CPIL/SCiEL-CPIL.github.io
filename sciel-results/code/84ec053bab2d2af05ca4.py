import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Computes a Batch-Aware Look-Ahead acquisition score.
    Simulates the next 5 evaluations greedily to estimate the expected
    improvement in f_best, updating f_best and sigma locally.
    """
    N = len(mu)
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Pre-allocate scores
    scores = np.zeros(N)
    
    # Base parameters for simulation
    f_best_init = f_best
    batch_size = 5
    
    # We iterate over each candidate as the "first" evaluation
    for i in range(N):
        # Check time budget
        if time.time() > deadline:
            break
            
        # Initialize simulation state
        mu_sim = mu.copy()
        sigma_sim = sigma.copy()
        f_best_sim = f_best_init
        
        # Mask for selected points
        selected = np.zeros(N, dtype=bool)
        selected[i] = True
        
        # Simulate the first evaluation (candidate i)
        # Assume it achieves its mean
        y_val = mu_sim[i]
        f_best_sim = min(f_best_sim, y_val)
        sigma_sim[i] = 0.0
        
        # Simulate subsequent evaluations (batch_size - 1 steps)
        for _ in range(batch_size - 1):
            if time.time() > deadline:
                break
                
            # Calculate EI for all points
            z = (f_best_sim - mu_sim) / sigma_sim
            ei = (f_best_sim - mu_sim) * norm.cdf(z) + sigma_sim * norm.pdf(z)
            
            # Mask out already selected points
            ei[selected] = -np.inf
            
            # Select the point with highest EI
            idx_next = np.argmax(ei)
            
            # If no valid point (all -inf), break
            if ei[idx_next] == -np.inf:
                break
                
            # Update state
            selected[idx_next] = True
            y_next = mu_sim[idx_next]
            f_best_sim = min(f_best_sim, y_next)
            sigma_sim[idx_next] = 0.0
            
            # "Update posterior variance locally to approximate information gain"
            # Heuristic: Global uncertainty reduction due to new data
            sigma_sim *= 0.99
            
        # Score for candidate i is the total improvement in f_best
        scores[i] = f_best_init - f_best_sim
        
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
