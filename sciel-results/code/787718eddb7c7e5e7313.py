import numpy as np
import time

_T0 = None
_DEADLINE = None


def _ensure_deadline():
    global _T0, _DEADLINE
    if _T0 is None:
        _T0 = time.perf_counter()
        _DEADLINE = _T0 + 295.0
    return _DEADLINE


def acquisition(mu, sigma, f_best):
    deadline = _ensure_deadline()
    elapsed = time.perf_counter() - _T0

    # If we are close to the deadline, fall back to a simple, fast exploitation strategy
    if elapsed > _DEADLINE - 5.0:
        return -mu

    n = mu.shape[0]

    # Safety clamps for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.minimum(sigma_safe, 1e4)

    # Estimate progress based on f_best
    # Goldstein-Price min is 3. Max in domain is around 1000-1200.
    f_max = 1000.0
    f_min = 3.0

    # p is 0 at start (f_best high), 1 at end (f_best low)
    p = (f_max - f_best) / (f_max - f_min)
    p = np.clip(p, 0.0, 1.0)

    # Structural Change: Use Expected Improvement (EI) as the primary acquisition function.
    # EI provides a natural balance between exploration and exploitation that adapts
    # to the current uncertainty (sigma) and potential improvement (f_best - mu).
    # This is a different modeling family from LCB/POI blends.

    # z-score for each candidate
    z = (f_best - mu) / sigma_safe

    # Clip z to avoid numerical overflow in exp/pdf calculations
    z = np.clip(z, -10, 10)

    # Calculate PDF and CDF using numpy for speed
    # pdf = (1/sqrt(2*pi)) * exp(-0.5 * z^2)
    # cdf = 0.5 * (1 + erf(z/sqrt(2)))
    
    # Using scipy.stats.norm is fine, but let's check if we can do it faster with numpy
    # For 2000 points, scipy is fast enough.
    
    from scipy.stats import norm
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf

    # EI is non-negative. Higher EI means better candidate.
    score = ei

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
