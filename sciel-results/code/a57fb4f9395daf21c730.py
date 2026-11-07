import numpy as np
import time

_LOG_START_TIME = time.time()
_DEADLINE = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling for minimization.
    """
    # Deadline guard
    if time.time() - _LOG_START_TIME > _DEADLINE - 5.0:
        # Fallback: return zeros to avoid timeout, though unlikely to trigger
        return np.zeros(len(mu))

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Draw a single sample from the posterior predictive distribution
    # z ~ N(0, 1)
    z = np.random.standard_normal(len(mu))
    
    # theta = mu + sigma * z
    theta = mu + sigma * z
    
    # Normalized regret: (theta - f_best) / sqrt(cost)
    # We want to minimize this, so we maximize its negative:
    # score = (f_best - theta) / sqrt(cost)
    sqrt_cost = np.sqrt(cost)
    
    score = (f_best - theta) / sqrt_cost
    
    # Handle numerical instabilities
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
