# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_DEADLINE = 330.0  # evaluator kills at 330s; use 325s for safety
_START_TIME = None


def _check_deadline():
    """Return True if we should stop and return current best."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > 325.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: float, lowest observed value
    - cost: array of shape (2000,), evaluation costs
    
    Returns:
    - array of shape (2000,), finite scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace any non-finite or invalid values
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Use cost-adjusted Lower Confidence Bound (LCB)
    # LCB = mu - beta * sigma
    # We want to minimize LCB, so acquisition = -LCB / cost
    # Higher acquisition = better candidate
    
    # Adaptive beta based on uncertainty and improvement potential
    # Use a fixed confidence parameter for stability
    beta = 2.0
    
    # Compute LCB
    lcb = mu - beta * sigma
    
    # Cost-adjusted score: we want low LCB and low cost
    # Score = -lcb / cost (higher is better)
    score = -lcb / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
