"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB: Balance exploration (sigma) and exploitation (mu)
    # normalized by evaluation cost.
    # Using a fixed kappa=1.0 for strong exploration in early stages,
    # which acts as a proxy for information gain per cost.
    kappa = 1.0
    ucb = mu + kappa * sigma
    return ucb / cost
# EVOLVE-BLOCK-END
