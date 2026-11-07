"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an exploration
    bonus to balance exploitation and exploration for 1D sphere minimization.
    
    Uses a modified EI with a small exploration term to avoid getting stuck
    in local optima, which is particularly useful for the sphere function.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: favor points with higher uncertainty
    # This helps escape local optima in the sphere function
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combined acquisition: EI + exploration bonus
    # The exploration term encourages visiting uncertain regions
    acq = ei + exploration_bonus
    
    # Ensure all values are finite
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    return acq
# EVOLVE-BLOCK-END
