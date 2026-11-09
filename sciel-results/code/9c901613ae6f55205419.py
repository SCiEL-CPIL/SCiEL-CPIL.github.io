"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) acquisition for minimization.
    # Score = mu - beta * sigma, so lower predicted mean with high uncertainty
    # gets a higher score. This is structurally different from EI (which uses
    # norm.cdf and norm.pdf) and POI (which uses only norm.cdf).
    # Higher score means more promising to evaluate next.
    beta = 2.0
    scores = mu - beta * sigma
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores
# EVOLVE-BLOCK-END
