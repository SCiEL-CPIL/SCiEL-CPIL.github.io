import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, keep 10s margin
    
    # Ensure sigma is positive and not too small for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Strategy: Additive Cost Penalty
    # Instead of dividing by cost (which heavily penalizes high-cost regions 
    # where the minimum likely resides), we subtract a small multiple of the cost.
    # This allows the algorithm to explore expensive regions if the expected 
    # improvement is sufficiently high, while still preferring cheaper points 
    # when improvements are comparable.
    
    # We tune the penalty factor. 
    # Cost ranges from 1.0 to 2.0.
    # A penalty of 0.1 * cost means a cost difference of 1.0 (max) 
    # affects the score by 0.1.
    # If f_best is around 10, EI can be several units. 
    # If f_best is 0.1, EI is small.
    # We want the cost to matter, but not dominate.
    
    penalty_factor = 0.1
    score = ei - penalty_factor * cost
    
    # Check time
    if time.time() > deadline:
         # Return a simple fallback if we are running out of time
         # Though this calculation is fast, we must be safe.
         return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
