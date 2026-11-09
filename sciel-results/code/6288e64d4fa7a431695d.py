"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 325.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Probability of Improvement (POI) per unit cost.
    
    POI replaces the continuous improvement magnitude of EI with a binary likelihood
    signal (probability that f(x) < f_best), which is more robust to noise and
    aggressively exploits high-confidence regions. Normalizing by cost directly
    optimizes the rate of improvement relative to expense.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coordinates).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: return neutral scores if we're running out of time
    if time.time() - _START_TIME > _TIMEOUT:
        return np.zeros(len(mu))
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values and safe divisions
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Probability of Improvement (POI)
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Ensure POI is in [0, 1] (numerical safety)
    poi = np.clip(poi, 0.0, 1.0)
    
    # Cost-aware scoring: POI / cost
    # This favors points that have high probability of improvement per unit cost
    score = poi / cost_safe
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
