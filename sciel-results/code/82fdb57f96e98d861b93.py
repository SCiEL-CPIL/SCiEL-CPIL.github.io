"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s kill


def _is_time_exceeded():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Args:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviation predictions
        f_best: scalar, best (lowest) observed value
        cost: array of shape (2000,), cost of evaluating each point
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we are running too long, return a safe fallback
    if _is_time_exceeded():
        # Fallback: prioritize low cost and high uncertainty, but keep it simple
        return -cost + sigma * 0.1

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to prevent extreme values in CDF/PDF which might lead to numerical issues
    # Although norm.cdf/pdf handle large values well, clipping is a safe practice
    z = np.clip(z, -50.0, 50.0)
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical noise might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost
    
    # Ensure all scores are finite
    # If any score is not finite (e.g., due to overflow), replace with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
