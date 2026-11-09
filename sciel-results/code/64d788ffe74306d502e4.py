"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We maximize the score: -mu + beta * sigma
    # This is equivalent to minimizing the Lower Confidence Bound: mu - beta * sigma
    beta = 2.0
    scores = -mu + beta * sigma
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return scores
# EVOLVE-BLOCK-END
