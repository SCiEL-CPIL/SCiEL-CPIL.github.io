"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to find the minimum, so we prefer points with low mean and high uncertainty.
    # The quantity to minimize is: mu - beta * sigma
    # The evaluator selects the point with the highest score.
    # Therefore, we return the negative of the quantity to minimize:
    # Score = -mu + beta * sigma
    
    beta = 2.0
    scores = -mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END
