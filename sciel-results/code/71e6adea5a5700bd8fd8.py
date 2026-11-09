"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to maximize the score, which corresponds to finding low function values.
    # score = -mu + beta * sigma
    # A higher beta encourages exploration of uncertain regions.
    # The Beale function has a sharp global minimum and flat valleys, so exploration is key.
    
    # Handle potential numerical issues with sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Beta controls the trade-off between exploitation (-mu) and exploration (sigma)
    beta = 2.5
    
    # The score is higher for lower mu and higher sigma
    scores = -mu + beta * sigma_safe
    
    return scores
# EVOLVE-BLOCK-END
