"""Rewritten cost-aware acquisition for levy 2d BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining EI and UCB exploration.
    
    mu: (2000,) predicted mean in objective units
    sigma: (2000,) predicted std in objective units
    f_best: lowest observed value
    cost: (2000,) cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns: (2000,) scores, higher = evaluate next
    """
    # Guard against degenerate sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration term: balance exploitation (low mu) with exploration (high sigma)
    # The beta coefficient controls exploration-exploitation tradeoff
    beta = 0.3
    ucb = -mu + beta * sigma_safe
    
    # Cost normalization: divide by cost to prefer cheaper points
    # Use a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Combine EI and UCB, both normalized by cost
    # EI is already non-negative; UCB can be negative for very good points
    # We want higher score = better
    # EI/cost rewards improvement per unit cost
    # UCB/cost rewards exploration per unit cost
    score = ei / cost_safe + 0.3 * (ucb / cost_safe)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
