"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Adaptive Upper Confidence Bound (UCB) for minimization.
    # Score = -mu + beta * sigma
    # Beta is high when f_best is large (exploration) and low when f_best is small (exploitation).
    # Scale factor 5.0 is chosen based on the magnitude of the Beale function values.
    beta = 0.5 + 2.5 * (1.0 - np.exp(-f_best / 5.0))
    scores = -mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END
