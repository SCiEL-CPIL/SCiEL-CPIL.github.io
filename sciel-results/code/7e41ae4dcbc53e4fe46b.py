"""Cost-aware UCB acquisition for styblinski tang 2d BO."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit
_SAFETY_MARGIN = 5.0  # seconds to spare


def _check_deadline():
    """Return True if we are close to the deadline."""
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for minimization.
    
    Uses Upper Confidence Bound (UCB) with a cost^2 penalty to strongly
    favor cheaper evaluations. For minimization, we maximize (beta*sigma - mu)/cost^2.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition values (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple heuristic
    if _check_deadline():
        # Fallback: simple UCB without cost normalization
        beta = 2.0
        scores = -mu + beta * sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: we want to find points where the lower bound
    # (mu - beta*sigma) is low, which means maximizing (beta*sigma - mu)
    # Use a moderate beta to balance exploration and exploitation
    beta = 2.0
    
    # UCB score for minimization (higher is better for selection)
    ucb = -mu + beta * sigma
    
    # Cost-aware scoring: divide by cost^2 to strongly penalize expensive evaluations
    # Cost range is [1, 2], so cost^2 ranges [1, 4]
    # This gives a 4x preference for the cheapest region over the most expensive
    cost_factor = np.power(np.maximum(cost, 1e-9), 2.0)
    
    scores = ucb / cost_factor
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
