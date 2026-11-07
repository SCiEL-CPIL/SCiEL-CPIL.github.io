"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Enhanced EI with adaptive exploration bonus.
    # Base EI for exploitation
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus: proportional to sigma but damped for large sigma
    # to avoid excessive exploration in far regions
    exploration_scale = 0.1
    sigma_clipped = np.clip(sigma, 0, 1.0)
    bonus = exploration_scale * sigma_clipped * np.exp(-sigma_clipped)
    
    return ei + bonus
# EVOLVE-BLOCK-END
