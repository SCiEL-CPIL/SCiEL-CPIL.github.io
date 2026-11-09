import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    # Leave a safety margin of 5 seconds
    return max(0.0, 300.0 - elapsed - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: array of shape (2000,) - GP mean
    sigma: array of shape (2000,) - GP standard deviation
    f_best: float - best observed value
    cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check time remaining
    remaining = _get_remaining_time()
    if remaining < 1.0:
        # Fallback: Prioritize low cost and low mu
        # This is a simple greedy fallback
        return (f_best - mu) / (cost + 1e-9)
    
    # Ensure inputs are arrays and float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Clip z-score to avoid numerical issues in norm.pdf/cdf
    # norm.pdf(>10) is effectively 0, norm.cdf(>10) is effectively 1
    z = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware acquisition: EI per unit cost
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
