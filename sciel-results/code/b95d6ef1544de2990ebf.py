"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition combining Expected Improvement (EI) with an exploration
    term, where the balance shifts based on the progress of the optimization.
    
    Early on, high uncertainty relative to improvement favors exploration to map
    the landscape. As convergence progresses (uncertainty decreases or improvement
    gap shrinks), the strategy shifts towards exploitation via EI.
    
    This combines the robustness of EI (inspiration 2) with the adaptive
    exploration-exploitation switching (inspiration 1) to handle the multimodal
    nature of Michalewicz 2D effectively within 30 evaluations.
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid overflow in norm.cdf/pdf
    z_clipped = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Compute Exploration Score (normalized sigma)
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    if sigma_max > sigma_min:
        exploration = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        exploration = np.ones_like(sigma_safe)
    
    # Determine progress indicator
    # If the best predicted value (min mu) is close to f_best, we are likely converged.
    # If there is a large gap and high uncertainty, we should explore more.
    min_mu = np.min(mu)
    gap = f_best - min_mu
    uncertainty_scale = np.max(sigma_safe)
    
    # Progress is 0 when we are far from convergence (large gap, high uncertainty)
    # Progress is 1 when we are close to convergence (small gap or low uncertainty)
    if uncertainty_scale > 1e-9:
        # Normalize the gap by the uncertainty scale
        # If gap is small relative to uncertainty, progress is high
        progress = np.clip(gap / (uncertainty_scale + 1e-9), 0.0, 1.0)
    else:
        progress = 1.0
    
    # We want to explore MORE when progress is LOW (early stage)
    # and EXPLOIT MORE when progress is HIGH (late stage)
    # 
    # Let's define the weight for exploration.
    # Early stage (progress ~ 0): exploration_weight should be high (e.g., 0.5)
    # Late stage (progress ~ 1): exploration_weight should be low (e.g., 0.0)
    # 
    # exploration_weight = 0.5 * (1.0 - progress)
    # exploitation_weight = 1.0 - exploration_weight
    
    exploration_weight = 0.5 * (1.0 - progress)
    exploitation_weight = 1.0 - exploration_weight
    
    # Normalize EI to [0, 1] range for blending with exploration score
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones_like(ei)
    
    # Final score is a weighted blend
    score = exploitation_weight * ei_norm + exploration_weight * exploration
    
    return score
# EVOLVE-BLOCK-END
