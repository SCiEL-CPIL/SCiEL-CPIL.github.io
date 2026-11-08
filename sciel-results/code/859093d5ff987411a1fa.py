"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost with exploration bonus.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Add a small exploration term that encourages evaluating uncertain points
    exploration = 0.1 * sigma / (sigma + 1e-9)
    return (ei + exploration) / cost
# EVOLVE-BLOCK-END
