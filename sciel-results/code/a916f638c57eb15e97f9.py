"""Bootstrap portfolio: MODEL-AND-OPTIMIZE approach for acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Pre-compute lookup tables for fast norm.cdf and norm.pdf evaluation
_LUT_SIZE = 1_000_001
_LUT_MIN = -10.0
_LUT_MAX = 10.0
_lut_z = np.linspace(_LUT_MIN, _LUT_MAX, _LUT_SIZE)
_lut_cdf = norm.cdf(_lut_z)
_lut_pdf = norm.pdf(_lut_z)


def _fast_norm_cdf(x):
    """Vectorized norm.cdf using interpolation for speed."""
    x = np.clip(x, _LUT_MIN, _LUT_MAX)
    idx = np.interp(x, _lut_z, _lut_cdf)
    return idx


def _fast_norm_pdf(x):
    """Vectorized norm.pdf using interpolation for speed."""
    x = np.clip(x, _LUT_MIN, _LUT_MAX)
    return np.interp(x, _lut_z, _lut_pdf)


def _ei(mu, sigma, f_best):
    """Expected Improvement."""
    sigma_safe = sigma + 1e-9
    z = (f_best - mu) / sigma_safe
    cdf = _fast_norm_cdf(z)
    pdf = _fast_norm_pdf(z)
    return (f_best - mu) * cdf + sigma_safe * pdf


def _lcB(mu, sigma, f_best, beta):
    """Lower Confidence Bound (for minimization)."""
    return mu - beta * sigma


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3D.
    
    Uses a time-adaptive hybrid of EI and LCB.
    """
    t_start = time.time()
    
    # Handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Check for non-finite values
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        valid = np.isfinite(mu) & np.isfinite(sigma)
        scores = np.zeros_like(mu)
        if np.any(valid):
            scores[valid] = _ei(mu[valid], sigma[valid], f_best)
        scores[~np.isfinite(scores)] = 0.0
        return scores
    
    # Calculate EI and LCB
    ei_scores = _ei(mu, sigma, f_best)
    
    # Adaptive beta for LCB
    # f_best starts around -1.0 to -3.0 and goes towards -3.86278
    # We want high exploration (high beta) when f_best is far from optimum,
    # and low exploration (low beta) when f_best is close.
    
    # Normalized progress: 0.0 at start (f_best = 0), 1.0 at optimum (f_best = -3.86278)
    # Hartmann 3D range is roughly [-3.86, 0]. 
    # Initial random points might be ~ -1.5.
    # Let's map f_best from [0, -3.86278] to [0, 1]
    # progress = (0 - f_best) / 3.86278
    # If f_best is 0, progress is 0. If f_best is -3.86, progress is 1.
    # However, f_best can be positive? Unlikely for Hartmann 3D in [0,1]^3, max is ~1.
    # Let's clamp f_best to be <= 0 for progress calculation to be safe, or just use the raw value.
    # Actually, Hartmann 3D global min is -3.86278.
    # Let's define a "closeness" metric.
    # If f_best > -1.0, we are very early.
    # If f_best < -3.0, we are late.
    
    # Linear interpolation of beta from 4.0 (early) to 1.5 (late)
    # Map f_best from [-1.0, -3.5] to [1.0, 0.0] for weight of exploration?
    # Or simpler:
    # beta = 2.5 + 1.5 * exp((f_best + 3.86278) / 0.5) ?
    # Let's use a smooth step.
    
    # Define early and late beta values
    beta_early = 4.0
    beta_late = 1.5
    
    # Estimate progress p in [0, 1]
    # p=0 when f_best is high (e.g., -1.0)
    # p=1 when f_best is low (e.g., -3.8)
    # Use a window of interest: [-1.0, -3.8]
    f_start = -1.0
    f_end = -3.86278
    span = f_end - f_start # negative
    
    # If f_best is better than f_start (more negative), progress increases
    # If f_best is worse than f_start (less negative), progress is 0
    # If f_best is better than f_end, progress is 1
    
    # Clip f_best to [f_end, f_start]
    f_clipped = np.clip(f_best, f_end, f_start)
    
    # p = (f_clipped - f_start) / (f_end - f_start)
    # If f_clipped == f_start (-1.0), p = 0
    # If f_clipped == f_end (-3.86), p = 1
    p = (f_clipped - f_start) / span
    
    # Ensure p is in [0, 1]
    p = np.clip(p, 0.0, 1.0)
    
    # Interpolate beta
    beta = beta_early + (beta_late - beta_early) * p
    
    # Calculate LCB scores
    lcb_scores = _lcB(mu, sigma, f_best, beta)
    
    # Hybrid: Normalize and combine
    # To combine EI and LCB, we should normalize them to comparable scales
    # EI is in objective units (improvement). LCB is in objective units (predicted value).
    # It's often better to use a weighted sum of normalized scores or just one.
    # Given the finding "Hybrid EI-LCB", let's try a weighted sum of normalized scores.
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei_scores)
    ei_max = np.max(ei_scores)
    if ei_max - ei_min < 1e-12:
        ei_norm = np.zeros_like(ei_scores)
    else:
        ei_norm = (ei_scores - ei_min) / (ei_max - ei_min)
        
    # Normalize LCB to [0, 1]
    # Note: LCB is mu - beta*sigma. Lower is better.
    # So high LCB score (in the sense of "goodness") means low LCB value.
    # We want to maximize the acquisition score.
    # So we should invert LCB: score = -LCB
    lcb_inverted = -lcb_scores
    
    lcb_min = np.min(lcb_inverted)
    lcb_max = np.max(lcb_inverted)
    if lcb_max - lcb_min < 1e-12:
        lcb_norm = np.zeros_like(lcb_inverted)
    else:
        lcb_norm = (lcb_inverted - lcb_min) / (lcb_max - lcb_min)
        
    # Weight: 
    # Early: more LCB (exploration)
    # Late: more EI (exploitation)
    # w_lcb = 1.0 - p
    # w_ei = p
    # But pure EI is often better than pure LCB late. 
    # Let's use a stronger shift.
    # w_ei = p**2 ? 
    # Let's try: w_ei = p, w_lcb = 1-p.
    
    w_ei = p
    w_lcb = 1.0 - p
    
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Final safety check
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
