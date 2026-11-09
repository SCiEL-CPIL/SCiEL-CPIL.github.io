"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) with a low beta to favor exploitation.
    # For minimization, we maximize (beta * sigma - mu).
    # A low beta ensures we primarily follow the mean (exploitation) while
    # allowing slight exploration in uncertain regions.
    beta = 0.1
    scores = beta * sigma - mu
    return scores
# EVOLVE-BLOCK-END
