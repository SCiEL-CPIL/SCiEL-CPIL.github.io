"""Cost-aware acquisition for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # 5s safety margin before 330s kill


def _safe_score(mu, sigma, f_best, cost):
    """Fallback: simple EI/cost with finite guarantees."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    score = ei / cost_safe
    return np.where(np.isfinite(score), score, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining Expected Improvement per unit cost
    with a moderate exploration bonus, adapted to the Griewank landscape.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: ensure finite inputs
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return _safe_score(np.nan_to_num(mu, nan=0.0),
                           np.nan_to_num(sigma, nan=1e-3),
                           float(np.nanmin(mu)) if np.any(np.isfinite(mu)) else 0.0,
                           np.nan_to_num(cost, nan=1.0))

    # Deadline guard
    if time.time() > _DEADLINE:
        return _safe_score(mu, sigma, f_best, cost)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI
    ei_cost = ei / cost_safe

    # Exploration bonus: normalize sigma to [0, 1]
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    if sigma_max > sigma_min:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma_safe)

    # Combine: 85% cost-aware EI + 15% exploration
    # The exploration term helps escape local minima in Griewank
    score = 0.85 * ei_cost + 0.15 * sigma_norm / cost_safe

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
