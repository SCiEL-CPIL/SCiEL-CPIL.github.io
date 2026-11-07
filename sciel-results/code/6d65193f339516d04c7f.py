"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement per unit cost.
    # PI is more robust to scale and focuses on the likelihood of finding a better point.
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    return pi / cost
# EVOLVE-BLOCK-END
