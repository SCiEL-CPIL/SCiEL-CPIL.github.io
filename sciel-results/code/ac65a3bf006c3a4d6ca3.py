"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Adaptive UCB
    # We want high exploration (high beta) when f_best is large (far from min).
    # We want low exploration (low beta) when f_best is small (close to min).
    # beta = beta_max * (1 - exp(-lambda * f_best))
    
    beta_max = 2.0
    lambda_decay = 5.0
    
    # Handle edge case where f_best is 0 or very small
    beta = beta_max * (1.0 - np.exp(-lambda_decay * f_best))
    
    # Ensure beta is not negative (f_best should be positive for Weierstrass)
    if beta < 0:
        beta = 0.0
        
    # UCB score: mu - beta * sigma (minimization)
    # Higher score means more promising (lower expected value)
    scores = mu - beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
