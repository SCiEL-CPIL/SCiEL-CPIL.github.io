"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization.
    # Higher score means more promising to evaluate.
    # We prefer low mu (likely low function value) and high sigma (uncertainty).
    kappa = 1.0
    score = -mu + kappa * sigma
    return score
# EVOLVE-BLOCK-END
