"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 330s limit, 40s safety margin


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _TIME_LIMIT:
        return np.zeros_like(mu)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware: divide by cost, but use a softer penalty
    # Use cost^0.5 instead of cost to avoid over-penalizing expensive regions
    scores = ei / np.sqrt(cost)

    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local minima in the acquisition landscape
    sigma_mean = np.mean(sigma)
    if sigma_mean > 0:
        exploration = 0.1 * sigma / sigma_mean
        scores = scores + exploration

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
