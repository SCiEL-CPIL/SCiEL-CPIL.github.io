"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize mu - beta * sigma.
    # The evaluator picks the highest score.
    # So score = -(mu - beta * sigma) = beta * sigma - mu.
    beta = 2.0
    scores = beta * sigma - mu
    return scores
# EVOLVE-BLOCK-END
