"""Cost-aware acquisition for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using UCB (Upper Confidence Bound) for minimization.
    Uses mu - beta * sigma as the optimistic lower bound, normalized by cost.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP std predictions
        f_best: scalar - best observed function value
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive beta: start with moderate exploration, decay as f_best improves
    # f_best for Griewank 2D ranges from ~0.5-1.0 initially to ~0
    # Use beta that scales with remaining uncertainty
    beta_base = 1.5  # Base exploration coefficient
    
    # Decay exploration as we get closer to the optimum
    # f_best close to 0 means we're doing well, reduce exploration
    decay = np.exp(-2.0 * f_best)
    beta = beta_base * (1.0 + decay)
    
    # UCB for minimization: we want to maximize (f_best - predicted_min)
    # where predicted_min = mu - beta * sigma (optimistic lower bound)
    # So score = (f_best - (mu - beta * sigma)) / cost
    #          = (f_best - mu + beta * sigma) / cost
    
    ucb_score = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # Add a small EI component to balance exploitation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_score = ei / cost_safe
    
    # Blend: primarily UCB, with a small EI component for stability
    alpha = 0.7  # Weight for UCB
    total_score = alpha * ucb_score + (1.0 - alpha) * ei_score
    
    # Ensure all values are finite
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    return total_score
# EVOLVE-BLOCK-END
