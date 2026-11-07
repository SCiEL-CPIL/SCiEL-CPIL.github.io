"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    This function computes a score for each candidate point based on the
    Upper Confidence Bound for minimization, normalized by the evaluation cost.
    Higher scores indicate candidates that are more likely to improve the
    best observed value per unit cost.
    
    Args:
        mu: Array of GP mean predictions, shape (N,).
        sigma: Array of GP standard deviations, shape (N,).
        f_best: Scalar, the lowest observed value so far.
        cost: Array of evaluation costs for each candidate, shape (N,).
        
    Returns:
        Array of finite scores, shape (N,).
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Exploration-exploitation trade-off parameter
    # A value of 2.5-3.0 is typically effective for short-horizon BO
    beta = 2.5
    
    # Calculate the Lower Confidence Bound (LCB) for minimization
    # Lower values are better.
    lcb = mu - beta * sigma
    
    # Calculate the potential improvement over the current best
    improvement = f_best - lcb
    
    # Normalize by cost to account for the budget constraint
    # Use a small epsilon to prevent division by zero
    safe_cost = np.maximum(cost, 1e-6)
    
    # The acquisition score is the improvement per unit cost
    scores = improvement / safe_cost
    
    # Ensure all scores are finite to satisfy the task contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
