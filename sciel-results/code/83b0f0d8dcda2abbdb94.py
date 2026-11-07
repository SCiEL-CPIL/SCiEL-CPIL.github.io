"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to maximize this score.
    # High score if mu is low (good) or sigma is high (uncertain).
    # kappa controls the trade-off.
    kappa = 1.0
    # Ensure sigma is non-negative and avoid division by zero if we were doing z-scores, 
    # but here we just add.
    ucb_scores = -mu + kappa * sigma
    return ucb_scores
# EVOLVE-BLOCK-END
