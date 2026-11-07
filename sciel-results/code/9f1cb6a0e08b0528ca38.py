"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to minimize the function, so we prefer candidates with low predicted mean (mu)
    # and high uncertainty (sigma).
    # The score is constructed such that higher values indicate higher priority.
    # Score = -mu + kappa * sigma
    # kappa controls the exploration/exploitation trade-off.
    # A value of 2.5 is a common choice for multimodal functions with limited evaluations.
    kappa = 2.5
    scores = -mu + kappa * sigma
    return scores
# EVOLVE-BLOCK-END
