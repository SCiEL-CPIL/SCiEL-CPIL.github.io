"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB)
    # beta controls the balance between exploration and exploitation.
    # A value around 2.0-3.0 is often effective for small budgets.
    beta = 2.5
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
