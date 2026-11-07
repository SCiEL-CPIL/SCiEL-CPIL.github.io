"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PoI) with a small exploration boost.
    # PoI favors broad coverage over peaking, which is better for multi-modal Branin.
    # Add a small UCB-like term to encourage exploration of high-uncertainty regions.
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    poi = norm.cdf(z)
    
    # Add a small exploration component: weight by sigma to explore uncertain areas
    # This helps escape local minima in the multi-modal Branin landscape
    ucb_weight = 0.1
    scores = poi + ucb_weight * sigma / (np.max(sigma) + eps)
    
    return scores
# EVOLVE-BLOCK-END
