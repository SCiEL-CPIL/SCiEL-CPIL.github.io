"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition combining Expected Improvement with a small exploration bonus.
    # EI per unit cost is the base; a tiny sigma term encourages exploring uncertain regions
    # without letting pure exploration dominate (which hurt in inspiration 2).
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Small exploration bonus to avoid getting stuck in local optima
    exploration_bonus = 0.05 * sigma_safe
    scores = (ei + exploration_bonus) / cost
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
