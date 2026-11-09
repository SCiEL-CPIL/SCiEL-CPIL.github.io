"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to maximize the score.
    # Score = mu - beta * sigma.
    # Low mu (good) and high sigma (uncertain) yield high score.
    beta = 2.0
    scores = mu - beta * sigma
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
