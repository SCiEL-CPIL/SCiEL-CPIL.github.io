"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to maximize the score.
    # Score = -mu + kappa * sigma
    # Lower mu is better (more negative), higher sigma is better (exploration).
    kappa = 1.5
    score = -mu + kappa * sigma
    return score
# EVOLVE-BLOCK-END
