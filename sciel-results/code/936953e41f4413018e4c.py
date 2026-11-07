"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: kill after 330s wall-clock; keep a safety margin.
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0  # seconds


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware expected improvement acquisition function.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: EI per unit cost
    # cost is in range [1, 2] since x is in [0,10]^4 and scaled to [0,1]
    # mean(x_scaled) is in [0,1], so cost is in [1,2]
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Return best found so far (which is just the computed scores)
        return scores
    
    return scores
# EVOLVE-BLOCK-END
