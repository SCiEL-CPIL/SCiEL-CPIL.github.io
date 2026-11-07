# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted UCB.
    
    Approach:
    - Uses Upper Confidence Bound (UCB) for exploration/exploitation balance.
    - Normalizes by cost to prefer cheaper evaluations that offer high potential improvement.
    - Score = (f_best - mu + beta * sigma) / cost
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Exploration coefficient
    # A value around 2-4 is typical for UCB in BO.
    # We can scale beta slightly if we want, but let's start with a constant.
    beta = 3.0
    
    # UCB for minimization:
    # We want to maximize the potential improvement: f_best - f(x)
    # The optimistic estimate of f(x) is mu - beta * sigma (lower is better)
    # So the optimistic improvement is f_best - (mu - beta * sigma) = f_best - mu + beta * sigma
    
    improvement = f_best - mu + beta * sigma_safe
    
    # Cost-aware score
    score = improvement / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
