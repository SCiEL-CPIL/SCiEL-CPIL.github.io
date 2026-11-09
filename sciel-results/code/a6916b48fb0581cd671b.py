"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware EI with mild exploration bonus.
    # EI: expected improvement per unit cost
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei_per_cost = ei / (cost + 1e-9)
    
    # Exploration bonus: reward high uncertainty regions
    # Normalize sigma to [0,1] range for stable weighting
    sigma_norm = sigma / (np.max(sigma) + 1e-9)
    # Blend: 90% exploitation (EI/cost), 10% exploration (uncertainty)
    # This helps escape local minima in 8D landscape
    score = 0.9 * ei_per_cost + 0.1 * sigma_norm
    
    return score
# EVOLVE-BLOCK-END
