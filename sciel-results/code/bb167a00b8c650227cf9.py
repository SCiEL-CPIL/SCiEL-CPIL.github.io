"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We minimize L(x) = mu(x) - beta * sigma(x).
    # The score to maximize is -L(x) = -mu(x) + beta * sigma(x).
    # Using beta = 2.5 to encourage exploration in the early stages.
    beta = 2.5
    scores = -mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END
