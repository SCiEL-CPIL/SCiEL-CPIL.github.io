"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Calculate Expected Improvement (EI)
    # Ensure sigma is not zero to avoid division issues, though scipy handles it well
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI formula for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Hybrid strategy: Add a small exploration term proportional to sigma
    # This encourages exploring high-uncertainty regions to escape local minima
    lambda_explore = 0.5
    scores = ei + lambda_explore * sigma
    
    return scores
# EVOLVE-BLOCK-END
