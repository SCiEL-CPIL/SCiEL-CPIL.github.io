"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses UCB (Upper Confidence Bound) per unit cost.
    
    Parameters:
    mu: array of shape (N,) - GP mean predictions
    sigma: array of shape (N,) - GP standard deviations
    f_best: float - lowest observed value so far
    cost: array of shape (N,) - evaluation costs
    
    Returns:
    array of shape (N,) - acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # UCB for minimization: lower is better, so we want to minimize mu - beta*sigma
    # Score should be higher when mu - beta*sigma is lower
    # So score = -(mu - beta*sigma) = -mu + beta*sigma
    beta = 2.0
    ucb_score = -mu + beta * sigma
    
    # Cost-aware score: UCB per unit cost
    score = ucb_score / cost
    
    # Ensure all outputs are finite to avoid invalid results
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
