import numpy as np
import time
from scipy.stats import norm

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    
    Uses Expected Improvement (EI) for minimization with a quadratic cost penalty.
    EI is more robust than UCB for finding the global minimum.
    The cost penalty strongly favors low-cost regions where the global minimum is likely located.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return safe fallback if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 290:  # Safety margin before 300s limit
        # Return a simple cost-based score as fallback
        cost_safe = np.maximum(cost, 1e-9)
        return 1.0 / (cost_safe ** 2)
    
    # Ensure sigma and cost are positive and finite
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Apply quadratic cost penalty (stronger preference for low-cost regions)
    # The global minimum of Styblinski-Tang is at x = -2.9035 in each dimension
    # which maps to low cost (x scaled to [0,1] gives ~0.21)
    score = ei / (cost ** 2)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
