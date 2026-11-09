"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization with exploration bonus
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration term to encourage visiting uncertain regions
    # This helps escape local minima in multi-modal functions like Branin
    exploration_bonus = 0.1 * sigma_safe
    
    return ei + exploration_bonus
# EVOLVE-BLOCK-END
