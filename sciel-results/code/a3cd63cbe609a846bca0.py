"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # Higher score means more promising.
    # We want to minimize, so we look for low mu and high sigma.
    # Score = -mu - kappa * sigma
    # Using kappa = 2.0 as a standard heuristic for exploration.
    kappa = 2.0
    return -mu - kappa * sigma
# EVOLVE-BLOCK-END
