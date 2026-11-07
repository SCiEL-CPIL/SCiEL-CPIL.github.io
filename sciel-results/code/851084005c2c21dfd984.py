"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We maximize the score, which corresponds to minimizing the function.
    # Lower mu is better (so -mu is high). Higher sigma is better for exploration (so +beta*sigma).
    beta = 1.0
    scores = -mu + beta * sigma
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    return scores
# EVOLVE-BLOCK-END
