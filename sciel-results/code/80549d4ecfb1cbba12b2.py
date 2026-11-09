"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Args:
        mu: Array of shape (N,) with predicted means.
        sigma: Array of shape (N,) with predicted standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with acquisition scores (higher is better).
    """
    # UCB for minimization: -mu + beta * sigma
    # beta controls the balance between exploitation and exploration.
    # beta=1 was found to be effective in similar contexts.
    beta = 1.0
    
    # Calculate UCB scores
    ucb_scores = -mu + beta * sigma
    
    # Make cost-aware by dividing by cost
    # Add a small epsilon to prevent division by zero, though min cost is 1.0
    epsilon = 1e-8
    cost_safe = cost + epsilon
    
    scores = ucb_scores / cost_safe
    
    # Ensure all scores are finite
    # If any value is NaN or Inf, replace with a very negative number
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
