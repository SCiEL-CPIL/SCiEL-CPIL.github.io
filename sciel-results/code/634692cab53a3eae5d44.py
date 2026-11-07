"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using probability-weighted improvement per unit cost.
    """
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # z-score for probability of improvement (minimization)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -15, 15)

    # Probability that this point improves over f_best
    p_improve = norm.cdf(z)

    # Expected improvement magnitude (conditional on improvement)
    # Using the standard EI formula but focusing on the improvement part
    improvement_magnitude = np.maximum(f_best - mu, 0.0)

    # Stochastic knapsack score: P(improve) * E[improvement] / cost
    # This directly maximizes expected regret reduction per unit cost
    score = p_improve * improvement_magnitude / cost_safe

    # Add a small exploration component to avoid getting stuck in local optima
    # Use uncertainty scaled by probability of being in a promising region
    exploration = sigma_safe * norm.pdf(z) / cost_safe

    # Combine: primarily the knapsack score, with slight exploration
    combined = 0.9 * score + 0.1 * exploration

    # Ensure all values are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)

    return combined
# EVOLVE-BLOCK-END
