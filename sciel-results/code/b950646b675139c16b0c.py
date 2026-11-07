"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement with small exploration constant.
    xi = 0.01 * (f_best - np.min(mu))
    z = (f_best - mu - xi) / (sigma + 1e-9)
    return (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
# EVOLVE-BLOCK-END
