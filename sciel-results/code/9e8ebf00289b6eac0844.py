"""Cost-aware acquisition for rastrigin 2d BO."""
import time
import numpy as np
from scipy.stats import norm

# Deadline guard constants
_DEADLINE_S = 320.0  # safety margin before 330s kill
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions for 2000 candidates
    sigma : array of shape (2000,)
        GP standard deviation predictions for 2000 candidates
    f_best : float
        Best (lowest) observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition values (higher = more desirable to evaluate next)
    """
    # Check deadline
    if _elapsed() > _DEADLINE_S:
        # Return a simple but valid fallback
        return np.zeros(2000, dtype=np.float64)
    
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute expected improvement
    z = (f_best - mu) / sigma
    # Clip z for numerical stability
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost
    # Add small regularization to prevent division issues
    scores = ei / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline again after computation
    if _elapsed() > _DEADLINE_S:
        # Return what we have, it's already computed
        return scores
    
    return scores
