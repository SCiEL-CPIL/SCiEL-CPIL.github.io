"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for the deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 330s limit

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) normalized by cost.
    EI provides a robust balance of exploitation and exploration.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Cost-aware acquisition: EI / cost
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If we're close to the deadline, return a simpler score to save time
    if _elapsed() > _DEADLINE_SECONDS:
        # Fall back to just EI/cost without any extra computation
        pass  # Already computed
    
    return scores
# EVOLVE-BLOCK-END
