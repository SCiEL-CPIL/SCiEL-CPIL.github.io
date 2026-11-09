"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to maximize the score to select the next point.
    # Since we minimize f, we look for low mu (good mean) and high sigma (uncertainty).
    # Score = -mu + beta * sigma
    # Beta is tuned based on the observation that Beta=2 improved over EI.
    # We increase Beta slightly to 2.5 to encourage more exploration in the
    # high-dimensional 6D space with a very small budget (25 evals).
    beta = 2.5
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
