"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB): mu - beta * sigma
    # Higher score means more promising for minimization.
    # Using a fixed beta to balance exploration and exploitation.
    beta = 2.5
    return mu - beta * sigma
# EVOLVE-BLOCK-END
