"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function that combines Expected Improvement (EI) with a small
    exploration bonus based on uncertainty, using z-clipping for numerical stability.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical overflow/underflow in norm.cdf and norm.pdf
    # norm.cdf(z) -> 0 for z << 0, -> 1 for z >> 0
    # norm.pdf(z) -> 0 for |z| >> 0
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Add a small exploration bonus to encourage sampling in uncertain regions
    # This helps escape local minima and explore the search space more broadly
    # The bonus is scaled by sigma to prioritize high-uncertainty regions
    exploration_bonus = 0.1 * sigma_safe
    
    # Combined score: EI + exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
