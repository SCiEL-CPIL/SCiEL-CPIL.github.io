"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement with an increased exploration term.
    # The previous reduction (0.9) led to local minima; increasing it (1.1) encourages
    # visiting high-variance regions to escape local basins in the multi-modal landscape.
    z = (f_best - mu) / (sigma + 1e-9)
    lambda_exploration = 1.1
    return (f_best - mu) * norm.cdf(z) + lambda_exploration * sigma * norm.pdf(z)
# EVOLVE-BLOCK-END
