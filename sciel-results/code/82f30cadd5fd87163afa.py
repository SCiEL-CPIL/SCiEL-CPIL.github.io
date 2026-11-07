"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound for minimization
    # Higher score means more likely to be evaluated next
    # LCB = -mu - kappa * sigma, so we return negative of that for "higher is better"
    kappa = 1.5
    lcb = mu + kappa * sigma
    # Negate so that lower LCB values get higher scores
    return -lcb
# EVOLVE-BLOCK-END
