"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (minimization): EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware weighting: penalize expensive evaluations (cost in [1, 2]).
    # Geometric penalty with gamma > 1 to strongly favor cheaper evaluations.
    gamma = 1.5
    cost_safe = np.maximum(cost, 1e-8)
    cost_factor = 1.0 / np.power(cost_safe, gamma)

    # Exploration bonus: small reward for low-uncertainty points to encourage
    # committing to a region rather than probing high-uncertainty areas.
    # This avoids redundant evaluations that waste the cost budget.
    exploration_beta = 0.05
    exploration_bonus = exploration_beta * sigma_safe

    # Combine: cost-aware EI + exploration bonus
    score = ei * cost_factor + exploration_bonus

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
