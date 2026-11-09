import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_START_TIME = time.time()
_LIMIT = 300.0


def _safe_min_max_normalize(arr):
    """Normalize array to [0, 1] range safely, handling edge cases."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_range = arr_max - arr_min
    if arr_range < 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - arr_min) / arr_range


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're running out of time, fall back to greedy
    elapsed = time.time() - _START_TIME
    if elapsed > _LIMIT - 10.0:
        # Greedy fallback: score = -mu (prefer lowest predicted mean)
        return -mu

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # ============================================================
    # 1. Expected Improvement (EI) for minimization
    # ============================================================
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # ============================================================
    # 2. Lower Confidence Bound (LCB) for minimization
    #    LCB = mu - beta * sigma  (lower is better for minimization)
    #    We want to MAXIMIZE the score, so use -LCB = -mu + beta*sigma
    #    Use adaptive beta based on number of evaluations estimated
    #    from the spread of mu and sigma
    # ============================================================
    # Adaptive beta: start exploratory, become more exploitative
    # Estimate "age" of the BO process from the distribution of sigma
    sigma_med = np.median(sigma)
    if sigma_med > 1e-6:
        # If sigma is still large, we're early -> explore more
        beta = 4.0
    elif sigma_med > 1e-4:
        beta = 2.5
    else:
        beta = 1.0

    lcb_score = -mu + beta * sigma

    # ============================================================
    # 3. Thompson Sampling proxy
    #    (f_best - mu) / sigma  -- high when improvement is likely
    #    relative to uncertainty
    # ============================================================
    ts_proxy = (f_best - mu) / (sigma + 1e-9)

    # ============================================================
    # 4. Probability of Improvement (PI)
    #    P(f < f_best | x) = Phi((f_best - mu) / sigma)
    # ============================================================
    pi = norm.cdf(z)

    # ============================================================
    # 5. Normalize all components to [0, 1]
    # ============================================================
    ei_norm = _safe_min_max_normalize(ei)
    lcb_norm = _safe_min_max_normalize(lcb_score)
    ts_norm = _safe_min_max_normalize(ts_proxy)
    pi_norm = _safe_min_max_normalize(pi)

    # ============================================================
    # 6. Weighted combination
    #    Use a non-linear blend that emphasizes the top candidates
    #    from each acquisition function
    # ============================================================
    # Base weights
    w_ei = 0.35
    w_lcb = 0.25
    w_ts = 0.25
    w_pi = 0.15

    raw_score = (w_ei * ei_norm +
                 w_lcb * lcb_norm +
                 w_ts * ts_norm +
                 w_pi * pi_norm)

    # Apply a mild power transformation to sharpen the distribution
    # (amplify differences between top candidates)
    raw_score = np.power(raw_score, 1.3)

    # Re-normalize after power transform
    score = _safe_min_max_normalize(raw_score)

    # Final safety: ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
