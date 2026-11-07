"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We maximize -mu + beta * sigma
    # beta = 1.0 is a standard choice
    beta = 1.0
    scores = -mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END
