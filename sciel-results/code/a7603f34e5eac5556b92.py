"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB): -mu + kappa * sigma
    # Higher score means more promising for next evaluation
    kappa = 1.5
    score = -mu + kappa * sigma
    return score
# EVOLVE-BLOCK-END
