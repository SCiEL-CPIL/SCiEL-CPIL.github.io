# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills the program after 330s wall-clock.
# We track elapsed time since module import and return early if we're close to the limit.
_T0 = time.time()
_DEADLINE = _T0 + 325.0  # 5s safety margin before 330s


def _check_deadline():
    """Return True if we should abort and return a fallback immediately."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: a structural blend of Expected Improvement (EI) and
    Lower Confidence Bound (LCB) for minimization, with an adaptive beta that
    shifts from exploration (early) to exploitation (late).

    mu, sigma : (2000,) arrays, sigma is the GP posterior std (objective units)
    f_best    : lowest observed objective value
    Returns 2000 finite scores; higher => evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-9)

    # Deadline guard
    if _check_deadline():
        # Fallback: pure exploitation (lowest mu) + tiny tiebreak
        scores = -mu + 1e-12 * np.arange(len(mu))
        return np.where(np.isfinite(scores), scores, 0.0)

    n = len(mu)

    # Adaptive beta: start with high exploration, decay toward exploitation.
    # Use the spread of mu relative to f_best as a proxy for how much we've learned.
    # When mu is close to f_best everywhere, we're in exploitation mode.
    # When there's high variance in mu, we should explore more.
    mu_range = np.ptp(mu) if n > 1 else 0.0
    # Normalize: if mu_range is small relative to sigma, we're confident -> exploit
    # If mu_range is large, we're uncertain -> explore
    sigma_mean = np.mean(sigma)
    if sigma_mean > 1e-12:
        uncertainty_ratio = mu_range / sigma_mean
    else:
        uncertainty_ratio = 0.0

    # Beta for LCB: high when uncertainty_ratio is high (early), low when low (late)
    # Clamp to [0.5, 5.0] to keep it reasonable
    beta = np.clip(0.5 + 1.5 * np.exp(-uncertainty_ratio), 0.5, 5.0)

    # LCB score for minimization: we want low values, so score = -(mu + beta * sigma)
    # Higher score = better (lower predicted value with confidence)
    lcb_scores = -(mu + beta * sigma)

    # EI for minimization: (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Normalize both to [0, 1] range for blending
    lcb_min, lcb_max = np.min(lcb_scores), np.max(lcb_scores)
    ei_min, ei_max = np.min(ei_scores), np.max(ei_scores)

    lcb_range = lcb_max - lcb_min
    ei_range = ei_max - ei_min

    if lcb_range > 1e-12:
        lcb_norm = (lcb_scores - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb_scores)

    if ei_range > 1e-12:
        ei_norm = (ei_scores - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei_scores)

    # Blend: weight LCB more when uncertainty is high, EI more when confident
    # w_lcb in [0.3, 0.7], w_ei = 1 - w_lcb
    w_lcb = np.clip(0.3 + 0.4 * np.exp(-uncertainty_ratio), 0.3, 0.7)
    w_ei = 1.0 - w_lcb

    scores = w_lcb * lcb_norm + w_ei * ei_norm

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Avoid all-equal scores
    if np.allclose(scores, scores[0]):
        scores = -mu + beta * sigma + 1e-12 * np.arange(n)
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
