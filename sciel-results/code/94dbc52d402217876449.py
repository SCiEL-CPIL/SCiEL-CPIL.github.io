"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    with an exponential cost penalty.
    
    This approach favors points that are both likely to improve the current best
    and are inexpensive to evaluate. The exponential penalty creates a sharp
    preference for cheaper regions, helping to escape local optima in expensive
    areas of the search space.
    """
    t0 = time.time()
    
    # Safety check for inputs
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.isfinite(f_best):
        return np.zeros(len(mu))
        
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1.0) # Cost is at least 1.0
    
    # 1. Calculate Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # 2. Apply Exponential Cost Penalty
    # The cost is in [1, 2]. We want to heavily penalize higher costs.
    # A lambda of 1.5 means a point with cost 2.0 is penalized by exp(-1.5) ~ 0.22
    # compared to a point with cost 1.0 (penalty 1.0).
    # This creates a strong bias towards the "cheap" part of the domain.
    # We can adapt lambda slightly based on f_best if needed, but fixed is robust.
    lambda_cost = 1.5
    cost_penalty = np.exp(-lambda_cost * cost_safe)
    
    # 3. Combine
    score = pi * cost_penalty
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() - t0 > 280:
        # Fallback: Simple PI / cost (less aggressive but safe)
        fallback = pi / cost_safe
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback
        
    return score
# EVOLVE-BLOCK-END
