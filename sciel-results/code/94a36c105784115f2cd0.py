"""Pareto-based multi-objective acquisition for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Dynamic weighted sum of normalized Expected Improvement (EI) and 
    normalized predictive uncertainty.
    
    The weight on exploitation (EI) increases as the maximum available EI decreases,
    indicating convergence towards the optimum. The weight on exploration 
    (uncertainty) decreases as convergence progresses.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior std dev in objective units
        f_best: float - best observed value so far (lowest)
    
    Returns:
        scores: array of shape (2000,) - higher scores mean evaluate next
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # EI is the expected improvement over f_best
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Predictive Uncertainty
    # We use sigma directly as the uncertainty measure
    uncertainty = sigma
    
    # 3. Determine Dynamic Weight for Exploitation
    # If the best possible EI is small, we are close to the optimum -> Exploit
    # If the best possible EI is large, we are far from the optimum -> Explore
    max_ei = np.max(ei)
    
    # Scale parameter C controls the transition speed.
    # For Hartmann 6D, a value around 0.1-0.5 is appropriate.
    # We use 0.2 as a baseline.
    C = 0.2
    
    # Weight on exploitation (EI)
    # w_exp is high when max_ei is low.
    # Using a squared term for a sharper but smooth transition.
    w_exploit = 1.0 / (1.0 + (max_ei / C) ** 2)
    
    # Weight on exploration (Uncertainty)
    w_explore = 1.0 - w_exploit
    
    # 4. Normalize Objectives
    # Normalize to [0, 1] range using the maximum value in the set.
    # This ensures the weighted sum is comparable across different scales.
    
    max_ei_val = np.max(ei)
    max_unc_val = np.max(uncertainty)
    
    # Avoid division by zero
    if max_ei_val < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / max_ei_val
        
    if max_unc_val < 1e-9:
        unc_norm = np.zeros_like(uncertainty)
    else:
        unc_norm = uncertainty / max_unc_val
        
    # 5. Combine Objectives
    score = w_exploit * ei_norm + w_explore * unc_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
