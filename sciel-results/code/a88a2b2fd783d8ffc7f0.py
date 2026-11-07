"""Rewritten program using Upper Confidence Bound (UCB) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # UCB for minimization
    # We maximize the score: -mu + beta * sigma
    # beta controls exploration. 
    # A common heuristic is beta = 1.0 to 2.0 for small problems.
    
    beta = 1.5
    
    # Calculate score
    # -mu encourages exploitation (low predicted mean)
    # beta * sigma encourages exploration (high uncertainty)
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
