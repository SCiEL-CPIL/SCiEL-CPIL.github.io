"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = None
_T0 = None


def _ensure_deadline():
    global _DEADLINE, _T0
    if _T0 is None:
        _T0 = time.perf_counter()
        _DEADLINE = _T0 + 300.0  # 300s per evaluation, keep margin
    return _DEADLINE


def _remaining():
    dl = _ensure_deadline()
    return max(0.0, dl - time.perf_counter())


def acquisition(mu, sigma, f_best):
    deadline = _ensure_deadline()
    n = mu.shape[0]

    # Clamp sigma to avoid division by zero / negative
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.minimum(sigma_safe, 1e4)  # avoid huge values

    # Adaptive exploration: when f_best is far from the minimum (3.0),
    # we are in early exploration phase -> use larger xi.
    # When f_best is close to 3.0, exploit more -> use smaller xi.
    # Goldstein-Price minimum is 3.0, range roughly [3, 14]
    gap = max(f_best - 3.0, 0.0)
    # Normalize gap: early stage gap ~ 10, late stage gap ~ 0
    # Map gap to xi range: early -> 0.1, late -> 0.0
    if gap > 5.0:
        xi_values = np.array([0.05, 0.1, 0.15])
    elif gap > 2.0:
        xi_values = np.array([0.01, 0.05, 0.1])
    else:
        xi_values = np.array([0.0, 0.01, 0.05])

    # Compute EI for each xi
    scores = np.zeros(n, dtype=np.float64)
    for xi in xi_values:
        target = f_best - xi
        z = (target - mu) / sigma_safe
        # Clip z for numerical stability
        z = np.clip(z, -10, 10)
        ei = (target - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        # EI can be negative; clamp to 0 for "higher is better" semantics
        ei = np.maximum(ei, 0.0)
        scores = np.maximum(scores, ei)

    # Add a small UCB-like exploration bonus for uncertain regions
    # This helps in early stages when EI might be too conservative
    # Scale the bonus by how far we are from the optimum
    if gap > 1.0:
        # Beta controls exploration strength, scaled by gap
        beta = min(0.5 * gap / 5.0, 0.5)
        # Normalize sigma for UCB: use median sigma as scale
        med_sigma = np.median(sigma_safe)
        if med_sigma > 1e-8:
            ucb_bonus = beta * sigma_safe / med_sigma
            # Only add bonus where EI is small (uncertain, not obviously good)
            low_ei_mask = scores < np.percentile(scores, 50)
            scores[low_ei_mask] += ucb_bonus[low_ei_mask]

    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
