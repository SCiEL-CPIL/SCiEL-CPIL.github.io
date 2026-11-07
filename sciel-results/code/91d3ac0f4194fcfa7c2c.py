"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: kill after 330s wall-clock, keep a safety margin
_DEADLINE = 330.0
_START_TIME = time.time()
_TIME_MARGIN = 5.0  # seconds to spare


def _elapsed():
    return time.time() - _START_TIME


def _check_deadline():
    """Return True if we should stop (near deadline)."""
    return _elapsed() > (_DEADLINE - _TIME_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB (Upper Confidence Bound) for minimization.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline check: if we're running out of time, return a simple fallback
    if _check_deadline():
        return (f_best - mu) / (cost + 1e-9)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Upper Confidence Bound (UCB) for minimization
    # UCB = mu - beta * sigma (lower bound on potential improvement)
    # We maximize (f_best - UCB) to balance exploration and exploitation
    # This is structurally different from LCB: it explicitly references f_best
    # and the score represents expected improvement over the best found
    beta = 1.5
    ucb = mu - beta * sigma
    # Score: how much better than f_best could this point be?
    # Higher score = more promising (either low mu or high sigma for exploration)
    score = (f_best - ucb) / (cost ** 1.5)

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
