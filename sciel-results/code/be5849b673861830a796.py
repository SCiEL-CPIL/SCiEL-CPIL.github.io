import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = None
_CALL_COUNT = 0

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _deadline_check(safety_margin=5.0):
    """Return True if we are safe to continue."""
    return _get_elapsed() < 330.0 - safety_margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of beale 2d.
    Uses an adaptive blend of Expected Improvement (EI) and Lower Confidence Bound (LCB).
    Early in the optimization, it favors LCB for exploration.
    Later, it favors EI for exploitation.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    t = _CALL_COUNT
    
    # Ensure we have finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Avoid division by zero in sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Compute Lower Confidence Bound (LCB) with adaptive beta
    # Start with high beta for exploration, decay over time
    # t is the step index (1 for 6th eval, etc.)
    beta_start = 4.0
    decay_rate = 0.15
    beta = beta_start / (1.0 + decay_rate * t)
    
    # LCB = mu - beta * sigma (lower is better)
    # Score = -LCB = -mu + beta * sigma (higher is better)
    score_lcb = -mu + beta * sigma_safe
    
    # Adaptive weighting
    # We want to shift from LCB to EI as t increases.
    # w_lcb starts high and decays.
    w_lcb = np.exp(-t / 15.0)
    w_ei = 1.0 - w_lcb
    
    # Normalize EI and LCB scores to [0, 1] for blending
    # Using min-max normalization on the current batch
    ei_min, ei_max = np.min(ei), np.max(ei)
    lcb_min, lcb_max = np.min(score_lcb), np.max(score_lcb)
    
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    
    ei_norm = (ei - ei_min) / ei_range
    lcb_norm = (score_lcb - lcb_min) / lcb_range
    
    # Blend
    score = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
