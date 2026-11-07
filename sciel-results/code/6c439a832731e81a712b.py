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
    Cost-aware acquisition function.
    
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

    # Lower Confidence Bound (LCB) for minimization
    # Structurally different from EI: does not depend on f_best directly.
    # Maximizes (beta * sigma - mu) to balance exploration (sigma) and exploitation (-mu).
    beta = 2.5
    lcb_score = beta * sigma - mu

    # Cost-aware: use quadratic penalty on cost to strongly prefer cheap evaluations
    # This allows more samples within the fixed budget of 30
    score = lcb_score / (cost ** 2)

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
