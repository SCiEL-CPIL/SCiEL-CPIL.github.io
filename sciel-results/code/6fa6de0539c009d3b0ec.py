"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of improvement per unit cost.
    # PI is more robust to high uncertainty than EI in low dimensions.
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    # Avoid division by zero or negative costs (cost is always >= 1)
    return pi / cost
# EVOLVE-BLOCK-END
