"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # UCB with cost normalization
    # We want to minimize f, so we look for low (mu - beta * sigma)
    # Score is higher for better candidates
    beta = 2.0
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-6)
    ucb = mu - beta * sigma
    return -ucb / cost_safe
# EVOLVE-BLOCK-END
