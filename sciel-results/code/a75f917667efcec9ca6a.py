# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition for minimizing a multimodal objective.
    # mu, sigma, cost: shape (2000,)
    # Higher returned score => evaluate next.

    # Guard: ensure finite inputs and positive sigma.
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # --- Expected Improvement (for minimization) ---
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Strict cost penalty: EI per unit cost.
    # This forces the algorithm to prioritize high-efficiency evaluations,
    # preventing budget waste on low-improvement points in the tight budget regime.
    score = ei / cost

    # Ensure finite output.
    score = np.where(np.isfinite(score), score, 0.0)
    # Ensure non-negative (higher = better).
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
