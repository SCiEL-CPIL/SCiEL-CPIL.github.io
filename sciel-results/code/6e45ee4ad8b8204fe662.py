"""Cost-aware acquisition function for Bayesian optimization (4 test functions)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.monotonic()
_DEADLINE = 325.0  # safety margin before 330s kill


def _elapsed():
    return time.monotonic() - _T0


def _deadline_ok():
    return _elapsed() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: UCB per unit cost, with numerical safety and deadline guard.
    """
    # Deadline guard: if we're running out of time, return a simple, fast score
    if not _deadline_ok():
        return -(mu + 0.5 * cost)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Defensive: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Avoid division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)

    # UCB (Upper Confidence Bound) for minimization:
    # We want to minimize, so "improvement" potential is f_best - mu + beta * sigma
    # Higher UCB value means more likely to find a better (lower) value.
    # The cell finding suggests beta=2 works well for exploration in cost-aware setting.
    beta = 2.0
    ucb = f_best - mu + beta * sigma_safe

    # Ensure UCB is non-negative (if mu > f_best, the "improvement" is negative, 
    # but we still might want to explore if sigma is high. However, strictly, 
    # if mu is already worse than f_best, EI is 0. UCB can be negative.
    # But typically we rank by score. Let's keep it as is, but ensure we don't 
    # accidentally penalize good points too much.
    # Actually, for minimization, the "regret" is f(x) - f_best.
    # UCB for minimization is often: mu - beta * sigma (we want low values).
    # Wait, let's re-verify the direction.
    # We want to pick x that minimizes f(x).
    # GP predicts mu, sigma.
    # We want to maximize the probability of finding f(x) < f_best.
    # Or maximize the expected improvement: EI = (f_best - mu) Phi(z) + sigma phi(z).
    # UCB approximation: f_best - mu + beta * sigma.
    # If mu is very low (good), f_best - mu is large positive.
    # If sigma is high (uncertain), beta * sigma adds to it.
    # So higher score = better. This is correct.
    
    # Cost-aware: UCB per unit cost
    score = ucb / cost_safe

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
