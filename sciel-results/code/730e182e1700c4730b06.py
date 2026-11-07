"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - scores: array of shape (2000,) - higher scores are preferred
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamically adjust kappa based on total cost consumed
    # Total cost is the sum of all costs evaluated so far
    # We estimate total cost from the cost array and the number of evaluations
    # But we don't have direct access to total cost, so we use a heuristic:
    # The cost array represents the cost of each candidate, and the total budget is 30
    # We can estimate progress by looking at the distribution of costs
    
    # A simpler approach: use the mean cost as a proxy for budget consumption
    # Early on, mean cost is moderate; later, if we're evaluating expensive points, mean cost increases
    # But actually, the cost is fixed per point, so the total cost accumulates over evaluations
    
    # Better approach: track the number of evaluations implicitly
    # Since we don't have that, we'll use a fixed schedule based on the cost distribution
    # The cost is 1 + mean(x scaled to [0,1]), so it ranges from 1 to 2
    # Total budget is 30, so we can do roughly 15-30 evaluations
    
    # For now, use a simple heuristic: kappa increases as we approach the budget
    # We estimate remaining budget by assuming average cost is 1.5
    # This is a rough estimate, but it should work reasonably well
    
    # Actually, let's use a simpler approach: use a fixed kappa that balances exploration and exploitation
    # For griewank function, which has many local minima, we want more exploration
    
    # Use LCB with cost adjustment
    # kappa = 2.0 is a good starting point for 2D problems
    kappa = 2.0
    
    # Lower Confidence Bound: mu - kappa * sigma
    lcb = mu - kappa * sigma_safe
    
    # Cost-aware score: LCB per unit cost
    scores = lcb / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
