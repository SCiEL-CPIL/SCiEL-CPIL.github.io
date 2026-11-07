"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # We maximize the score, which corresponds to minimizing the Lower Confidence Bound.
    # LCB = mu - beta * sigma
    # We want to maximize the potential for low values, so we use:
    # Score = beta * sigma - mu
    # This encourages exploration (high sigma) and exploitation (low mu).
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-8)
    
    # Exploration parameter
    # A value around 1.0 to 2.0 is typical. 
    # Given the multimodal nature of Styblinski-Tang and limited evaluations,
    # we lean towards exploration.
    beta = 2.0
    
    score = beta * sigma - mu
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
