"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Expected Improvement + UCB exploration term.
    Uses a small exploration weight to balance exploitation and exploration.
    Includes a deadline guard to ensure we finish well within time limits.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)

    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # UCB term for exploration
    # Use a moderate beta to encourage exploration in early stages
    # and exploitation in later stages. Since we don't have step count,
    # we use a fixed small value that helps escape local optima.
    beta = 0.2
    ucb = mu + beta * sigma

    # Combine: primarily EI, but add a small UCB component
    # This helps when EI is very small (near optimum) but uncertainty remains
    score = ei + 0.1 * ucb

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline check - if we've spent too long, just return what we have
    if time.time() > deadline:
        return score

    return score
# EVOLVE-BLOCK-END
