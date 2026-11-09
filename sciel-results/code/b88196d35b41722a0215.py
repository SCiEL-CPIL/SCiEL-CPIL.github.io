import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 10.0  # 10s safety margin per call, total 50s for 5 calls, well under 300s
    
    # Calculate z-score
    # sigma is standard deviation
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Calculate Cost
    # cost is passed as an argument? 
    # "mu, sigma and cost have shape (2000,)"
    # Yes, cost is passed in.
    
    # Linear Cost Penalty
    # Normalize cost to [0, 1] range for consistent penalty scale
    # Min cost is 1, Max cost is 2 (approx)
    # Let's use the actual min/max of the passed cost array for robustness
    c_min = np.min(cost)
    c_max = np.max(cost)
    c_range = c_max - c_min
    if c_range < 1e-9:
        c_norm = np.zeros_like(cost)
    else:
        c_norm = (cost - c_min) / c_range
        
    # Score = PI - lambda * c_norm
    # Tuning lambda: 
    # If lambda is too high, we stay in cheap corners. 
    # If lambda is too low, we ignore cost.
    # The "linear cost penalty" clue suggests this form.
    # Let's try lambda = 1.0.
    score = pi - 1.0 * c_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline check (though this is very fast)
    if time.time() > deadline:
        # Fallback: return pure PI if something went wrong, though unlikely
        return pi
    
    return score
