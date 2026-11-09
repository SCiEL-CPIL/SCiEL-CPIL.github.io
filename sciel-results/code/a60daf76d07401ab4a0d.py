"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # Objective: minimize mu - beta * sigma.
    # Score to maximize: beta * sigma - mu.
    
    # Beta parameter controls exploration vs exploitation.
    # Higher beta = more exploration.
    # For Branin2D with limited evaluations, moderate exploration is key.
    beta = 2.0
    
    scores = beta * sigma - mu
    
    # Ensure scores are finite
    # Replace non-finite values with a very low score to avoid selection
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
