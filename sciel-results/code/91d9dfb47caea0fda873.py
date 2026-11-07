"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d) using LCB."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive and handle numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute the standard LCB: mu - kappa * sigma
    # We use a dynamic exploration parameter kappa that adapts to remaining budget
    # The idea: as we use more of our budget, we should explore less
    
    # Estimate remaining budget fraction
    # Total budget is 30, initial cost from 5 uniform points
    # cost here represents the cost of each new candidate
    # We'll use a simple heuristic based on the current cost values
    
    # For cosine8 in 8D, we want a moderate exploration rate
    # kappa controls the balance between exploitation and exploration
    # A larger kappa means more exploration
    
    # Dynamic kappa: start with higher exploration, decrease as budget depletes
    # We estimate how much budget we've used by looking at typical costs
    # The cost is 1 + mean(x scaled to [0,1]), so it's in [1, 2]
    
    # Use a fixed but well-tuned kappa for 8D cosine8
    # In high dimensions, variance can be inflated, so we want to be careful
    # kappa = 1.0 to 2.0 is typical; we'll use a value that balances exploration
    
    kappa = 1.5
    
    # LCB score: lower is better for the objective, so we want to minimize mu - kappa*sigma
    # For acquisition (higher is better), we negate: -mu + kappa*sigma
    lcb_score = -mu + kappa * sigma
    
    # Cost normalization: divide by cost to prefer cheaper evaluations
    # This ensures we get more value per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    normalized_score = lcb_score / cost_safe
    
    # Ensure all scores are finite
    normalized_score = np.where(np.isfinite(normalized_score), normalized_score, -1e10)
    
    return normalized_score
# EVOLVE-BLOCK-END
