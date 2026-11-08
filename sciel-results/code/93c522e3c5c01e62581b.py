# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = None
_START_TIME = None

def _ensure_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 295.0  # 5s safety margin before 300s
    return _DEADLINE

def _time_left():
    return max(0.0, _ensure_deadline() - time.time())

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    deadline = _ensure_deadline()
    
    # Handle edge cases
    if len(mu) == 0:
        return np.array([])
    
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-12)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: normalize by cost
    scores = ei / (cost + 1e-12)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
