"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Adaptive Expected Improvement with exploration bonus
    # Use a small exploration parameter that decreases over time
    # Since we don't have access to step count, use sigma magnitude as proxy
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small UCB-like exploration term to encourage diversity early on
    # The coefficient is small to not overwhelm exploitation
    beta = 0.1
    ucb_term = beta * sigma
    
    # Combine: weight EI heavily, add exploration
    score = ei + ucb_term
    
    return score
# EVOLVE-BLOCK-END
