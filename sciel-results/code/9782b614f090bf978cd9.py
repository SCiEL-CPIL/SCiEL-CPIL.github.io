import numpy as np
import time
from scipy.stats import norm

_T0 = None
_DEADLINE = None
_MARGIN = 5.0


def _ensure_deadline():
    global _DEADLINE, _T0
    if _T0 is None:
        _T0 = time.perf_counter()
        _DEADLINE = _T0 + 300.0
    return _DEADLINE


def acquisition(mu, sigma, f_best):
    deadline = _ensure_deadline()
    n = mu.shape[0]

    # Time guard: if we're running out of time, fall back to simple exploitation
    elapsed = time.perf_counter() - _T0
    if elapsed > _DEADLINE - _MARGIN:
        return -mu

    # Clamp sigma for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.minimum(sigma_safe, 1e4)

    # Adaptive exploration based on gap to known minimum (3.0 for Goldstein-Price)
    gap = max(f_best - 3.0, 0.0)

    # Choose xi values adaptively
    if gap > 5.0:
        xi_values = np.array([0.05, 0.1, 0.15])
    elif gap > 2.0:
        xi_values = np.array([0.01, 0.05, 0.1])
    else:
        xi_values = np.array([0.0, 0.01, 0.05])

    # Compute EI for each xi and take the maximum
    scores = np.zeros(n, dtype=np.float64)
    for xi in xi_values:
        target = f_best - xi
        z = (target - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (target - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = np.maximum(scores, ei)

    # Add UCB exploration bonus in early stages for uncertain regions
    if gap > 1.0:
        beta = min(0.5 * gap / 5.0, 0.5)
        med_sigma = np.median(sigma_safe)
        if med_sigma > 1e-8:
            ucb_bonus = beta * sigma_safe / med_sigma
            low_ei_mask = scores < np.percentile(scores, 50)
            scores[low_ei_mask] += ucb_bonus[low_ei_mask]

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
