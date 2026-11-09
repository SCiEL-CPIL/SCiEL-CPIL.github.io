"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses an adaptive combination of EI and LCB with a time-guard.
    """
    start_time = time.time()
    deadline = start_time + 2.0  # Safety margin well within 300s limit

    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Compute LCB (Lower Confidence Bound) for minimization
    # mu - beta * sigma; lower is better
    beta = 2.0
    lcb = mu - beta * sigma_safe
    lcb = np.where(np.isfinite(lcb), lcb, 0.0)

    # Check deadline before normalization
    if time.time() - start_time > deadline * 0.5:
        # Return EI-based scores quickly
        scores = np.where(np.isfinite(ei), ei, 0.0)
        return scores

    # Normalize EI to [0, 1]
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)

    # Normalize LCB to [0, 1] (lower LCB is better, so flip)
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-12:
        # For minimization, lower LCB is better, so we want high score for low LCB
        lcb_norm = (lcb_max - lcb) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb)

    # Adaptive weighting based on uncertainty level
    sigma_mean = np.mean(sigma_safe)
    f_scale = max(abs(f_best), 1e-6)
    uncertainty_ratio = np.clip(sigma_mean / f_scale, 0.0, 2.0)

    # When uncertain (high uncertainty_ratio), weight LCB more for exploration
    # When certain (low uncertainty_ratio), weight EI more for exploitation
    w_lcb = np.clip(0.3 + 0.4 * uncertainty_ratio, 0.2, 0.7)
    w_ei = 1.0 - w_lcb

    # Combine
    scores = w_ei * ei_norm + w_lcb * lcb_norm

    # Scale back to a meaningful magnitude
    scores = scores * ei_range + ei_min

    # Final cleanup
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
