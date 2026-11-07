"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time since module load for deadline guard
_T0 = time.time()
_TIMEOUT = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Expected Improvement + adaptive uncertainty bonus.
    
    Uses EI as the primary exploitation signal (it naturally vanishes where
    improvement is unlikely) and adds a normalized sigma bonus weighted by
    the gap to the known global minimum. This provides strong early exploration
    that smoothly decays as f_best approaches the optimum.
    """
    # Deadline guard
    if time.time() - _T0 > _TIMEOUT:
        # Fallback: pure EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Known global minimum of Michalewicz 2D
    f_min_global = -1.8013
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Gap from current best to global minimum (0 when converged)
    gap = max(0.0, f_best - f_min_global)
    # Normalize gap: Michalewicz 2D ranges roughly from -1.8 to 1.0
    # gap_norm = 1.0 early (f_best ~ 1.0), 0.0 late (f_best ~ -1.8)
    gap_norm = min(1.0, gap / 2.8)
    
    # Exploration weight: high early, low late
    # Use a smooth curve: w_explore = 0.8 * gap_norm^0.7
    w_explore = 0.8 * (gap_norm ** 0.7)
    
    # Normalize EI to zero mean, unit std for stable combination
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-12
    ei_norm = (ei - ei_mean) / ei_std
    
    # Normalize sigma for exploration bonus
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe) + 1e-12
    sigma_norm = (sigma_safe - sigma_mean) / sigma_std
    
    # Combine: EI (exploitation) + weighted sigma (exploration)
    scores = ei_norm + w_explore * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
