"""Cost-aware acquisition for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: keep a safety margin before the 300s per-evaluation limit.
# The acquisition function is called repeatedly in a loop; we track total elapsed
# time across calls and, if we are approaching the limit, fall back to a
# deterministic, cheap heuristic so we never exceed the budget.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 10s safety margin


def _elapsed_ok():
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) objective value observed so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate, in the same units as the budget.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure finite outputs.
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)

    # If we are close to the deadline, use a simple, fast heuristic.
    if not _elapsed_ok():
        # Fallback: minimize predicted value per cost, with a small exploration bonus.
        scores = -mu / cost_safe
        # Add a tiny exploration term to avoid degenerate ties.
        scores += 1e-6 * sigma_safe / cost_safe
        return np.where(np.isfinite(scores), scores, -1e30)

    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    # This is positive only when mu < f_best.
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with very large values.
    z = np.clip(z, -50, 50)

    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Cost-aware: divide by cost.
    scores = ei / cost_safe

    # Add a small exploration bonus weighted by uncertainty relative to cost,
    # to encourage evaluating uncertain regions early.
    # Use a square-root of sigma to temper the exploration bonus.
    exploration = 0.1 * np.sqrt(sigma_safe) / cost_safe
    scores = scores + exploration

    # Ensure finite output.
    scores = np.where(np.isfinite(scores), scores, -1e30)

    return scores
# EVOLVE-BLOCK-END
