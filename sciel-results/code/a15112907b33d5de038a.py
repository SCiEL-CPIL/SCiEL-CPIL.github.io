"""Rewritten acquisition: cost-aware EI blended with UCB, with a deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level state for deadline guard
_DEADLINE = None
_DEADLINE_LIMIT = 320.0  # safety margin below 330s kill
_CALL_COUNT = 0


def _ensure_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.monotonic() + _DEADLINE_LIMIT
    return _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function blending EI and UCB."""
    global _CALL_COUNT
    _CALL_COUNT += 1

    deadline = _ensure_deadline()
    remaining = deadline - time.monotonic()

    # Safety: if almost out of time, return simple EI/cost quickly
    if remaining < 0.5:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / np.maximum(cost, 1e-12)
        return np.where(np.isfinite(score), score, 0.0)

    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # UCB term: explore where uncertainty is high
    # Beta controls exploration; moderate value for multimodal functions
    beta = 1.5
    ucb = beta * sigma_safe

    # Blend: weighted combination, normalize so scale is comparable
    # EI is in objective units, UCB is in objective units (sigma is std dev)
    # Cost-normalize both
    ei_cost = ei / cost_safe
    ucb_cost = ucb / cost_safe

    # Blend with a weight that favors exploitation but allows exploration
    # w_ei high means more exploitation
    w_ei = 0.7
    w_ucb = 0.3
    score = w_ei * ei_cost + w_ucb * ucb_cost

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
