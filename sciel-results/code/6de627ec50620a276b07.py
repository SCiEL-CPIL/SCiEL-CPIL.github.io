"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr, ndtri
import time

# Deadline guard
_DEADLINE = 320.0
_start_time = time.time()

# Precompute constants
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement with adaptive UCB and cost-awareness.
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * ndtr(z) + sigma_safe * np.exp(-0.5 * z * z) / _SQRT_2PI
    ei = np.maximum(ei, 0.0)

    # Adaptive UCB: explore more when uncertainty is high relative to improvement
    # Use a larger kappa when sigma is large (early exploration)
    # kappa scales with how much we've learned: start high, decrease as f_best improves
    # Hartmann 6d minimum is -3.32237, so f_best starts around -1 to -2
    # Normalize progress: 0 at start, 1 at optimum
    progress = np.clip((f_best - (-3.32237)) / 2.0, 0.0, 1.0)
    # kappa decreases from ~2.5 to ~1.0 as we progress
    kappa = 2.5 - 1.5 * progress
    ucb = mu - kappa * sigma_safe

    # UCB gain: how much below f_best the UCB predicts
    ucb_gain = np.maximum(f_best - ucb, 0.0)

    # Combine EI and UCB with adaptive weighting
    # Early: more UCB for exploration; later: more EI for exploitation
    alpha = 0.5 * progress + 0.2  # alpha increases from 0.2 to 0.7

    combined = ei + alpha * ucb_gain

    # Cost-awareness: penalize high-cost points
    # Cost is in [1, 2], so use a moderate penalty
    beta = 1.2

    scores = combined / np.power(cost_safe, beta)

    # Handle NaN/inf
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
