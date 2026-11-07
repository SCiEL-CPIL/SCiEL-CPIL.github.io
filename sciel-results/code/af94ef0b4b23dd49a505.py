"""Rewritten program using Expected Improvement (EI) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # Score = mu - beta * sigma
    # Higher score means lower expected value and/or higher uncertainty
    # f_best is not explicitly used in standard UCB, but kept for signature compatibility
    
    beta = 1.5
    
    score = mu - beta * sigma
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
