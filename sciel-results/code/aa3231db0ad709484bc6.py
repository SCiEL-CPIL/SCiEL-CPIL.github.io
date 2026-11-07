"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 320.0

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the limit, return a fast fallback
    if _elapsed() > _DEADLINE - 5.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-Effective Expected Improvement: EI / cost
    # This directly optimizes the trade-off between improvement and budget consumption.
    ce_ei = ei / cost_safe

    # Add a small exploration bonus for regions with high uncertainty
    # but low EI (to avoid getting stuck in exploitation).
    # The bonus is scaled by 1/cost to remain cost-aware.
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe / cost_safe

    score = ce_ei + exploration_bonus

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
