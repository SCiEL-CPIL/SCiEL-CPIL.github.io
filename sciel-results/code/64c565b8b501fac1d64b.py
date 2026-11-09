"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization.
    # Score is higher for points with lower expected minimum.
    kappa = 1.0
    return -mu + kappa * sigma
# EVOLVE-BLOCK-END
