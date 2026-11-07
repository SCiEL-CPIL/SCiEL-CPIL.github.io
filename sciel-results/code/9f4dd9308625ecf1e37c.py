"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're getting close to the 330s limit, return a safe default
    if time.time() - _start_time > 320:
        # Return scores that favor points with low mean (exploitative fallback)
        return -mu

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Add a small exploration bonus based on uncertainty to avoid getting stuck
    # This helps escape local optima in the early stages
    # Scale the exploration term so it's meaningful but not dominant
    exploration = 0.1 * sigma_safe

    # Combine EI with exploration
    scores = ei + exploration

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
