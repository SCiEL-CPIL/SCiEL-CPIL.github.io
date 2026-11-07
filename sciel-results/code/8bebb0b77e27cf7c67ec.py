"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition: blends pure exploration (max sigma) with Expected Improvement
    based on how close the best prediction is to f_best relative to overall uncertainty.
    
    Early on (high uncertainty relative to improvement), favor exploration to map the
    landscape. As convergence progresses, shift toward exploitation via EI.
    """
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration score: raw sigma (normalized)
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    if sigma_max > sigma_min:
        exploration = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        exploration = np.ones_like(sigma_safe)
    
    # Progress indicator: how much improvement we've seen relative to total uncertainty
    # If mu is close to f_best everywhere, we're converged -> exploit
    # If there's still large gap and high sigma -> explore
    gap = f_best - np.min(mu)
    uncertainty_scale = np.max(sigma_safe)
    
    if uncertainty_scale > 1e-9:
        progress = np.clip(gap / (uncertainty_scale + 1e-9), 0.0, 1.0)
    else:
        progress = 1.0
    
    # Blend weight: when progress is low (early), weight exploration more
    # When progress is high (converged), weight exploitation more
    # Use a smooth transition
    exploration_weight = 0.5 * (1.0 - progress)
    exploitation_weight = 1.0 - exploration_weight
    
    # Normalize EI to [0,1] range for blending
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones_like(ei)
    
    score = exploitation_weight * ei_norm + exploration_weight * exploration
    
    return score
# EVOLVE-BLOCK-END
