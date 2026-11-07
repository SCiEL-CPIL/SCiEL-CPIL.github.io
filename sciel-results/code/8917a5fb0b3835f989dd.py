"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # We maximize the score: -mu + beta * sigma.
    # This balances exploitation (low mu) and exploration (high sigma).
    # A beta of 2.0 is a standard heuristic for 1D problems.
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite to satisfy the contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
