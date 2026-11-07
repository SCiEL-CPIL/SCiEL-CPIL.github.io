# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Adjusted Lower Confidence Bound (LCB).
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - best observed value so far (unused in LCB)
    cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-8)
    
    # Avoid numerical issues in sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Exploration-exploitation trade-off parameter
    # Fixed beta is a standard heuristic for LCB
    beta = 2.0
    
    # For minimization, we use the Lower Confidence Bound: mu - beta * sigma
    # We want to maximize the acquisition score, so we use -LCB: -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-adjusted score
    scores = lcb_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
