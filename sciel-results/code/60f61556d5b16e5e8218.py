"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate Expected Improvement (EI)
    # mu: (2000,)
    # sigma: (2000,)
    # f_best: scalar
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI formula
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration bonus
    # The knowledge suggests adding an uncertainty-based exploration bonus improved the score.
    # We add a term proportional to sigma.
    # Beta controls the trade-off. 0.5 is a moderate exploration weight.
    beta = 0.5
    exploration_bonus = beta * sigma
    
    return ei + exploration_bonus
# EVOLVE-BLOCK-END
