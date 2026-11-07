"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the acquisition score for each candidate point.
    Uses Expected Improvement (EI) plus a linear exploration bonus.
    
    Args:
        mu: (N,) array of posterior means.
        sigma: (N,) array of posterior standard deviations.
        f_best: float, the best (lowest) observed value so far.
        
    Returns:
        (N,) array of acquisition scores. Higher is better.
    """
    # Ensure sigma is positive to avoid division issues, though GP should provide >0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If mu is much smaller than f_best, z is large positive.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using scipy.stats.norm.cdf and pdf
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Add exploration bonus
    # The insight suggests 0.1 * sigma improves performance.
    exploration_bonus = 0.1 * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure scores are finite
    # If any nan/inf occur, replace with a very low value to avoid selecting them
    # Or just clip. But EI + sigma should be finite for finite inputs.
    # If sigma is 0, z is inf, cdf is 1, pdf is 0. EI = f_best - mu.
    # This is handled by sigma_safe.
    
    return scores
# EVOLVE-BLOCK-END
