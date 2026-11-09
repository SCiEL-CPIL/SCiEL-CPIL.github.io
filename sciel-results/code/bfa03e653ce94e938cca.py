"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blends Expected Improvement (exploitation) with
    a lower-confidence-bound / Thompson-like exploration term.
    
    The blend weight shifts from exploration-heavy early to exploitation-heavy
    late, estimated from how close f_best is to the known minimum (0 for Beale).
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Best observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    t0 = time.time()
    
    # Safety margin: if we're running long, just return EI quickly
    deadline = 300.0  # seconds total budget per evaluation step
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # --- Expected Improvement ---
    z_ei = (f_best - mu) / sigma_safe
    # Handle cases where z is very large positive (certain improvement)
    z_ei_clipped = np.clip(z_ei, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_ei_clipped) + sigma_safe * norm.pdf(z_ei_clipped)
    
    # --- Lower Confidence Bound (exploration) ---
    # LCB = mu - kappa * sigma; we want high score for low predicted value + uncertainty
    # Since we're minimizing, lower mu and higher sigma should get higher score
    # Score component: -mu + beta * sigma
    # Normalize: use a beta that adapts
    
    # Estimate progress: f_best starts high (Beale max ~170+) and goes to 0
    # Use a simple heuristic: if f_best is still large, explore more
    # Beale function min is 0, typical max in [-4.5,4.5]^2 is around 170
    # Use f_best relative to initial scale
    f_scale = max(f_best, 1.0)  # avoid log(0)
    
    # Adaptive beta: larger when f_best is large (early), smaller when f_best is small (late)
    # beta ~ sqrt(log(1 + 1/f_best)) or similar
    # Simpler: beta = 1.0 when f_best > 10, tapering to 0.3 when f_best < 0.1
    if f_best > 50:
        beta = 1.5
    elif f_best > 10:
        beta = 1.0
    elif f_best > 1:
        beta = 0.5
    elif f_best > 0.1:
        beta = 0.25
    else:
        beta = 0.1
    
    # LCB-based score: we want to maximize this
    # Since lower mu is better, and higher sigma encourages exploration:
    lcb_score = -mu + beta * sigma_safe
    
    # --- Blend ---
    # Weight: early on, give more weight to LCB; late on, more to EI
    # Use a sigmoid-like transition based on f_best
    # When f_best is large (early), w_lcb is high; when small (late), w_lcb is low
    # Normalize EI and LCB to comparable scales before blending
    
    # Scale EI: EI values are in objective units, typically small
    # Scale LCB: also in objective units
    
    # Compute relative weights
    # w = 1 / (1 + exp(k * (f_best - threshold)))
    # When f_best >> threshold, w ~ 1 (more LCB)
    # When f_best << threshold, w ~ 0 (more EI)
    threshold = 5.0
    k = 0.5
    w_lcb = 1.0 / (1.0 + np.exp(k * (f_best - threshold)))
    w_ei = 1.0 - w_lcb
    
    # Normalize both to [0, 1] range for fair blending
    # EI: min could be 0, max varies
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-12:
        lcb_norm = (lcb_score - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb_score)
    
    # Blended score
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    elapsed = time.time() - t0
    if elapsed > deadline - 5:
        # Fallback: just return EI to be safe
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
