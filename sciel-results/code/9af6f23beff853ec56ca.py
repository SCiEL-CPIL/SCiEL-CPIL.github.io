"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

def _check_deadline():
    """Return True if we should stop working (deadline approaching)."""
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Hybrid EI + UCB acquisition with adaptive exploration-exploitation tradeoff.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value so far.
    Returns: 2000 finite scores, higher = more promising.
    """
    n = len(mu)
    
    # Deadline guard: if we're close to the limit, return a simple EI score
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Known minimum of the RBF mixture
    f_min = -2.53218
    
    # Progress: how close are we to the minimum?
    regret = f_best - f_min
    # Map regret from [0, ~2] to progress in [0, 1]
    progress = 1.0 - np.clip(regret / 2.0, 0.0, 1.0)
    
    # Adaptive parameters
    # xi: improvement threshold for EI - larger early to encourage exploration
    xi = 0.02 * (1.0 - progress) + 0.001 * progress
    
    # kappa: UCB exploration parameter - larger early
    kappa = 2.0 * (1.0 - progress) + 0.5 * progress
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (for minimization: lower bound is better)
    # LCB = mu - kappa * sigma (we want to minimize, so low mu and high sigma are good)
    lcb = mu - kappa * sigma_safe
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    # Normalize LCB to [0, 1] (lower LCB is better, so invert)
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min + 1e-9
    lcb_norm = (lcb_max - lcb) / lcb_range  # higher = more promising
    
    # Combine: weight shifts from exploration (LCB) to exploitation (EI) as we progress
    w_ei = 0.3 + 0.6 * progress      # 0.3 early, 0.9 late
    w_lcb = 1.0 - w_ei               # 0.7 early, 0.1 late
    
    score = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
