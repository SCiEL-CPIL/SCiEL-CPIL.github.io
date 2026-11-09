"""Bayesian Regret Minimization acquisition for cost-aware BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Bayesian Regret Minimization.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).

    Returns
    -------
    np.ndarray, shape (2000,)
        Finite scores; higher = prefer to evaluate next.
    """
    # Explicit deadline guard
    if _deadline_exceeded():
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu / (sigma_safe + 1e-9) + np.log(sigma_safe + 1.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Clamp sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)

    # Bayesian Regret Minimization:
    # E[min(f(x), f_best)] ≈ f_best - 0.5 * max(0, f_best - mu - 0.5*sigma)
    # We want to MINIMIZE regret, so we MAXIMIZE the reduction in expected regret.
    # Expected regret reduction = f_best - E[min(f(x), f_best)]
    #                            = 0.5 * max(0, f_best - mu - 0.5*sigma)
    
    regret_reduction = 0.5 * np.maximum(0.0, f_best - mu - 0.5 * sigma_safe)
    regret_reduction = np.where(np.isfinite(regret_reduction), regret_reduction, 0.0)

    # Cost-aware: normalize by cost to get regret reduction per unit cost
    score = regret_reduction / cost

    # Add a small exploration bonus for high uncertainty regions
    # This helps when regret_reduction is zero (mu is already below f_best)
    sigma_max = np.max(sigma_safe) if len(sigma_safe) > 0 else 1.0
    if sigma_max > 0:
        exploration_bonus = 0.05 * (sigma_safe / sigma_max) / cost
    else:
        exploration_bonus = np.zeros_like(mu)

    total_score = score + exploration_bonus

    # Ensure finiteness
    total_score = np.where(np.isfinite(total_score), total_score, -1e10)

    return total_score
# EVOLVE-BLOCK-END
