"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound for minimization:
    # We want to minimize f, so we prefer points with low mu and high sigma (exploration).
    # LCB = mu - beta * sigma
    # Since we return scores where higher = evaluate next, we return -LCB = -mu + beta * sigma
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
