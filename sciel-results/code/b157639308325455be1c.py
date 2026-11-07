"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize f(x).
    # The lower bound of the confidence interval is mu - beta * sigma.
    # We select the point with the lowest lower bound.
    # The acquisition function should return higher scores for better points.
    # So we return - (mu - beta * sigma) = beta * sigma - mu.
    
    # Beta controls the exploration-exploitation trade-off.
    # A value of 2.0 is a standard heuristic for moderate exploration.
    beta = 2.0
    
    # Ensure sigma is non-negative (it should be by definition, but just in case)
    sigma = np.maximum(sigma, 0.0)
    
    scores = beta * sigma - mu
    
    return scores
# EVOLVE-BLOCK-END
