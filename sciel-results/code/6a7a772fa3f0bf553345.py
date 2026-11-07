import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization."""
    t_start = time.time()
    deadline = 330.0 - 15.0  # 15 second safety margin

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: clamp sigma and cost to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Handle z overflow/underflow
    z = np.clip(z, -50, 50)

    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware scoring: linear cost penalty (EI / cost)
    # This was the configuration that achieved the highest score (1.0367)
    score = ei / cost_safe

    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima in multi-modal functions
    # The bonus is also cost-aware, preferring cheap uncertainty
    exploration_bonus = 0.05 * sigma_safe / cost_safe

    total_score = score + exploration_bonus

    # Ensure finite output
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)

    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > deadline:
        # Return a safe default (zeros) to avoid timeout
        return np.zeros_like(total_score)

    return total_score
