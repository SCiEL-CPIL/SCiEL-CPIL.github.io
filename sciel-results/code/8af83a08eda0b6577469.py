"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound for minimization
    # Maximize -mu + beta * sigma
    beta = 1.0
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
