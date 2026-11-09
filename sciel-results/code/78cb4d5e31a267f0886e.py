"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to maximize the score to pick the next point.
    # Low mean and high uncertainty are desirable.
    beta = 0.5
    return mu - beta * sigma
# EVOLVE-BLOCK-END
