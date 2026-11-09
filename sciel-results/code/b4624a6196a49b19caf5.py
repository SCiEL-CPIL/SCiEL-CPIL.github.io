"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) acquisition function
    # score = mu + beta * sigma
    # Higher score means more promising (for minimization, we want low mu but high sigma for exploration)
    # Since we're minimizing, mu is the predicted value (lower is better)
    # sigma is the uncertainty (standard deviation)
    # We want to balance: low mu (exploitation) and high sigma (exploration)
    beta = 1.5
    score = mu + beta * sigma
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
