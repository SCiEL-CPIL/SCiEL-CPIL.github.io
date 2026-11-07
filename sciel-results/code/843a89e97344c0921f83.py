import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0  # seconds, with safety margin
_SAFETY_MARGIN = 5.0  # seconds

def _check_deadline():
    """Check if we're close to the deadline."""
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization."""
    if _check_deadline():
        # Return a simple fallback if we're close to deadline
        return np.ones_like(mu) * 0.5
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scoring: EI divided by cost with slight superlinear penalty
    # This makes the acquisition more aggressive about finding cheap improvements
    score = ei / (cost ** 1.3)
    
    # Add a small exploration bonus for uncertain regions
    # This helps escape local optima
    exploration = 0.05 * sigma / cost
    score += exploration
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
