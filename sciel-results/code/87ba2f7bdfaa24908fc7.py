"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We maximize the score: -mu + beta * sigma
    # This balances exploitation (low mu) and exploration (high sigma).
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    # The problem statement guarantees mu and sigma are from a valid GP,
    # but we add a check just in case to avoid invalid results.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
