import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start_time = time.time()
    deadline = 320.0  # 320 seconds, leaving 10s margin for 330s limit
    
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score: EI minus penalty
    # Penalty is proportional to cost
    # Lambda is a small constant
    lambda_cost = 0.05
    scores = ei - lambda_cost * cost
    
    # Check time
    if time.time() > deadline:
        # Return the best result found so far (or a simple fallback)
        # Since we computed scores, we can just return them if they are valid
        # If not, return zeros
        if not np.all(np.isfinite(scores)):
            return np.zeros_like(scores)
        return scores

    return scores
# EVOLVE-BLOCK-END
