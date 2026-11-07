"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Shifted Expected Improvement (HVI approximation)
    # Reference point is f_best + delta
    # This encourages larger jumps in improvement
    
    delta = 0.5
    ref = f_best + delta
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (ref - mu) / sigma_safe
    
    # Expected Improvement formula
    ei = (ref - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
