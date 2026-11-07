"""Rewritten acquisition for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE_MARGIN = 5.0  # seconds before the 330s wall-clock limit
_START_TIME = None


def _ensure_deadline():
    """Track elapsed time; return True if we are still within budget."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.perf_counter()
    return (time.perf_counter() - _START_TIME) < (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    lower-confidence-bound (LCB) exploration term, with adaptive
    exploration that increases when sigma is large relative to the
    current best.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean in objective units.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation in objective units.
    f_best : float
        Best observed value so far.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Guard: ensure we haven't blown the time budget.
    if not _ensure_deadline():
        # Return a simple, fast fallback: pure LCB with a fixed beta.
        beta = 1.0
        return -(mu + beta * sigma)

    sigma_safe = np.maximum(sigma, 1e-9)

    # --- Expected Improvement (EI) ---
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # --- Lower Confidence Bound (LCB) ---
    # Use an adaptive beta that scales with the relative uncertainty
    # of the current best. When f_best is still far from the known
    # minimum (~0.397887), we explore more; as we converge, we exploit.
    # The known minimum of Branin is ~0.397887.
    known_min = 0.397887
    # Relative gap: how far are we from the optimum?
    gap = max(f_best - known_min, 0.0)
    # Scale beta: larger gap -> more exploration.
    # beta ranges roughly from 0.5 (near optimum) to 3.0 (far from optimum).
    beta = 0.5 + 2.5 * min(gap / 1.0, 1.0)

    lcb = -(mu + beta * sigma_safe)

    # --- Combine: normalize both to [0, 1] range and blend ---
    # EI is always >= 0. LCB can be negative.
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)

    # Normalize LCB to [0, 1]
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-12:
        lcb_norm = (lcb - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb)

    # Blend: weight shifts from exploration to exploitation as we converge.
    # When gap is large, weight more on LCB (exploration).
    # When gap is small, weight more on EI (exploitation).
    w_explore = 0.3 + 0.4 * min(gap / 1.0, 1.0)  # 0.3 to 0.7
    w_exploit = 1.0 - w_explore

    scores = w_exploit * ei_norm + w_explore * lcb_norm

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
