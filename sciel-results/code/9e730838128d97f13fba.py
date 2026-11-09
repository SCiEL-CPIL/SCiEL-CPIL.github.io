"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score, which corresponds to minimizing the function.
    # The lower confidence bound is L(x) = mu(x) - beta * sigma(x).
    # We want to pick x with the lowest L(x).
    # So the acquisition score (to be maximized) is S(x) = -L(x) = -mu(x) + beta * sigma(x).
    
    beta = 1.5
    scores = beta * sigma - mu
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
