"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to minimize the function, so we look for points with low mean
    # and high uncertainty. The acquisition function is:
    # score = mu - beta * sigma
    # We maximize this score.
    
    # Beta controls the exploration/exploitation trade-off.
    # A value around 2.5 to 5.0 is common. 
    # Since we don't know the iteration count, we use a fixed value.
    beta = 2.5
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute UCB score
    ucb_score = mu - beta * sigma_safe
    
    # Ensure finite values
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    return ucb_score
# EVOLVE-BLOCK-END
