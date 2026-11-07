"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Hybrid EI + adaptive exploration acquisition for Hartmann 6D minimization.
    
    Structural change from pure LCB: uses Expected Improvement (EI) as the
    primary exploitation signal, blended with a sigma-weighted exploration
    term. The blend ratio is scheduled over the 30-evaluation budget to
    start exploratory and end exploitative.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: return pure exploitation to avoid timeout
    if time.time() - _start_time > 280:
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Ensure valid inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Expected Improvement for minimization ---
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Normalize EI to have zero mean and unit std for stable blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-12:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = ei - ei_mean
    
    # --- Adaptive exploration term ---
    # Use sigma directly but normalize it
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    if sigma_std > 1e-12:
        sigma_norm = (sigma_safe - sigma_mean) / sigma_std
    else:
        sigma_norm = sigma_safe - sigma_mean
    
    # --- Schedule: exploration weight decreases over the budget ---
    # 30 total evaluations, 5 initial random, so 25 acquisition-driven steps
    # _call_count goes from 1 to 25 (approximately)
    # Start with high exploration, end with low exploration
    t = min(_call_count, 25) / 25.0  # 0 to 1 over the budget
    # Exploration weight: starts at 0.7, ends at 0.1
    w_explore = 0.7 - 0.6 * t
    w_exploit = 1.0 - w_explore
    
    # --- Blend ---
    scores = w_exploit * ei_norm + w_explore * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
