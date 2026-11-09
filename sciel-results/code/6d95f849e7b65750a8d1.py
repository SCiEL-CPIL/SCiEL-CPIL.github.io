"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 330 - 10  # Safety margin of 10 seconds


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best observed function value so far
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        # Fallback: simple cost-aware greedy (minimize mu, penalize cost)
        # Use a stable computation
        safe_cost = np.maximum(cost, 1e-8)
        return -mu / safe_cost
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Replace non-finite values with safe defaults
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
        cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # Compute EI
    ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but numerical issues can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
