"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Use UCB (Upper Confidence Bound) for exploration/exploitation balance
    # In minimization, we maximize -mu + kappa * sigma
    # kappa controls exploration vs exploitation
    # Small length scale (0.2) suggests local GP, UCB helps escape local optima
    kappa = 2.0 
    ucb = -mu + kappa * sigma
    # Divide by cost to be cost-aware
    # Ensure cost is not zero (it is >= 1)
    return ucb / cost
# EVOLVE-BLOCK-END
