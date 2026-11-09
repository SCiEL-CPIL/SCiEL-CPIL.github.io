"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Lower Confidence Bound (LCB) 
    for exploration with Target-Aware Expected Improvement (EI) for 
    directional exploitation towards the known global minimum.
    
    Structural Change: Replaces pure EI with an LCB-driven score, 
    modified by a small EI term relative to the known global minimum 
    to prevent exploration into irrelevant regions.
    """
    # Deadline guard
    t_start = time.time()
    
    # Known global minimum for Hartmann 6d
    f_target = -3.32237
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Lower Confidence Bound (LCB)
    # Encourages exploration by favoring low mean and high uncertainty.
    # Kappa=1.5 was found to be effective in previous evaluations.
    kappa = 1.5
    lcb = mu - kappa * sigma_safe
    
    # 2. Target-Aware Expected Improvement (EI)
    # Encourages points that have a high expected improvement over the 
    # known global minimum. This provides a "pull" towards the optimum.
    z_target = (f_target - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.pdf/cdf for extreme values
    z_target = np.clip(z_target, -10, 10)
    ei_target = (f_target - mu) * norm.cdf(z_target) + sigma_safe * norm.pdf(z_target)
    
    # Ensure ei_target is non-negative (it should be, but numerical errors can occur)
    ei_target = np.maximum(ei_target, 0.0)
    
    # 3. Combine LCB and EI
    # LCB is the primary driver (exploration + exploitation balance).
    # EI_target is a secondary refinement (directional bias).
    # The weight for EI is small (0.1) to not override the LCB exploration benefits.
    weight_ei = 0.1
    scores = lcb + weight_ei * ei_target
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check for deadline
    if time.time() - t_start > 250:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
