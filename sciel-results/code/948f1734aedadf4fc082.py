"""Acquisition function for Bayesian optimization of Beale 2D.

Uses a hybrid acquisition: Expected Improvement (EI) for exploitation
plus a Lower Confidence Bound (LCB) term for exploration. The LCB
coefficient is adapted based on the gap between f_best and the best
predicted mean, giving more exploration early and more exploitation
late. A time-guard ensures we return valid scores well within the
330 s wall-clock limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track a global start time for the deadline guard
_T0 = time.perf_counter()
_DEADLINE = 325.0  # seconds; 5 s safety margin before 330 s kill


def _time_left():
    return max(0.0, _DEADLINE - (time.perf_counter() - _T0))


def _expected_improvement(mu, sigma, f_best):
    """Standard Expected Improvement for minimization."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Where sigma is ~0, EI should be max(0, f_best - mu)
    ei = np.where(sigma < eps, np.maximum(f_best - mu, 0.0), ei)
    return np.where(np.isfinite(ei), ei, 0.0)


def _lower_confidence_bound(mu, sigma, kappa):
    """LCB for minimization: mu - kappa * sigma."""
    return mu - kappa * sigma


def acquisition(mu, sigma, f_best):
    """Return scores for 2000 candidate points; higher score = better."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # Deadline guard: if we're running out of time, return a simple EI score
    if _time_left() < 5.0:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(sigma < 1e-12, np.maximum(f_best - mu, 0.0), ei)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei

    # Numerical safety
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute EI
    ei = _expected_improvement(mu, sigma, f_best)

    # Compute LCB with adaptive kappa
    # Progress proxy: gap between f_best and best predicted mean
    mu_min = float(np.min(mu))
    gap = f_best - mu_min  # positive if predictions suggest improvement possible

    # Typical uncertainty
    sigma_med = float(np.median(sigma))
    if sigma_med < 1e-10:
        sigma_med = 1e-10

    # Adaptive kappa: larger when gap is large (early), smaller when gap is small (late)
    # Use a smooth function: kappa ~ base * (1 + gap / (gap + sigma_scale))
    # When gap >> sigma_scale: kappa ~ 2*base
    # When gap << sigma_scale: kappa ~ base
    sigma_scale = 0.5 * sigma_med
    progress = gap / (abs(gap) + sigma_scale + 1e-12)
    progress = float(np.clip(progress, 0.0, 1.0))

    # Base kappa for LCB
    kappa_base = 0.8
    kappa = kappa_base * (0.5 + progress)  # ranges from 0.4*kappa_base to 1.5*kappa_base

    lcb = _lower_confidence_bound(mu, sigma, kappa)

    # Normalize EI and LCB to comparable scales
    # EI is always >= 0; LCB can be negative
    # We want a weighted combination where both contribute meaningfully
    # Scale LCB so its typical magnitude is comparable to EI
    ei_med = float(np.median(ei[ei > 0])) if np.any(ei > 0) else 1.0
    if ei_med < 1e-10:
        ei_med = 1e-10

    # Normalize LCB: shift so min is 0, then scale
    lcb_shifted = lcb - np.min(lcb)
    lcb_scale = float(np.median(lcb_shifted[lcb_shifted > 0])) if np.any(lcb_shifted > 0) else 1.0
    if lcb_scale < 1e-10:
        lcb_scale = 1e-10

    # Combine: weighted sum of normalized EI and LCB
    # Weight shifts from exploration (LCB) to exploitation (EI) as progress increases
    w_ei = 0.7 + 0.3 * progress  # 0.7 to 1.0
    w_lcb = 1.0 - w_ei           # 0.3 to 0.0

    score = w_ei * (ei / ei_med) + w_lcb * (lcb_shifted / lcb_scale)

    # Final safety: ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
