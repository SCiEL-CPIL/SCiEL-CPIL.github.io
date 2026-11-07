"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement per unit cost (with softened cost penalty)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Use square root of cost to soften the penalty for expensive points
    # This allows exploration of moderately expensive regions if they promise better objective values
    scores = ei / np.sqrt(cost + 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
