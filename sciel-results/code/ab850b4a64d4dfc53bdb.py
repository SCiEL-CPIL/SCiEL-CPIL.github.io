"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Variance-Gated Expected Improvement (VEI).
    
    Scales standard EI by (sigma / sigma_max)^gamma to penalize exploitation
    in low-uncertainty regions and force exploration in high-uncertainty areas.
    This helps escape local minima in Branin's multimodal landscape.
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Variance gating: scale by (sigma / sigma_max)^gamma
    # This penalizes regions where the GP is already certain (low sigma)
    # and rewards regions where the GP is uncertain (high sigma)
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        gamma = 1.5
        variance_gate = (sigma_safe / sigma_max) ** gamma
    else:
        variance_gate = np.ones_like(sigma_safe)
    
    # Combine: EI * variance_gate
    # But we want to maintain some exploitation capability, so blend
    # with a small weight on raw EI to avoid pure exploration
    score = ei * variance_gate + 0.05 * ei
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
