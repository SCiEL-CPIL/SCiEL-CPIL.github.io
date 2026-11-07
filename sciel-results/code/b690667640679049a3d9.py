"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per square root of cost.
    # Reduces the penalty for high-cost regions, allowing exploration
    # of areas where the minimum might be located despite higher cost.
    sigma = np.maximum(sigma, 1e-6)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.pdf/cdf for extreme values
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei / np.sqrt(cost)
# EVOLVE-BLOCK-END
