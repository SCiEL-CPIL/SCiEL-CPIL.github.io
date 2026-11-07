"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # Higher score means more promising (lower expected value or higher uncertainty).
    # Score = -mu + beta * sigma
    # Using a fixed beta to balance exploration and exploitation.
    beta = 2.0
    scores = -mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END
