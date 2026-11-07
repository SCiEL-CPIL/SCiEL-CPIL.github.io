# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    Uses a linear combination of exploitation (f_best - mu) and exploration (beta * sigma),
    normalized by the evaluation cost.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # UCB parameter for exploration
    # Higher beta encourages exploration of high-uncertainty regions
    beta = 1.5

    # Exploitation term: how much better the mean is than the best found
    # Exploration term: how uncertain we are
    # We want to maximize: f_best - (mu - beta * sigma)
    # Which is: f_best - mu + beta * sigma
    ucb_term = f_best - mu + beta * sigma_safe

    # Cost-aware score
    scores = ucb_term / cost_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
