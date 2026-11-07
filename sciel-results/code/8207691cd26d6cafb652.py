"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 320s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    GP-UCB with budget-scaled exploration coefficient.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - lowest observed function value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: just use negative mean (exploitation only)
        return -mu
    
    # Budget-scaled exploration coefficient
    # Total budget is 30, we need to estimate how much budget we've used
    # We don't have direct access to total cost used, but we can use the fact
    # that cost array values are in [1, 2] and we've done some evaluations
    
    # Strategy: Use a decay based on the ratio of current cost to max cost
    # As we approach the budget limit, we want to be more exploitative
    
    # Estimate progress: we know we start with 5 points and budget is 30
    # Average cost per point is ~1.5, so we can do about 20 total evaluations
    # Let's use a simple heuristic: scale beta by remaining budget fraction
    
    # Since we don't know exact budget used, use a conservative approach:
    # Start with high exploration, decay linearly to low exploration
    # Use the mean of the cost array as a proxy for how expensive evaluations are
    
    # A simpler approach: use a fixed decay schedule based on the fact that
    # we're doing at most ~20 evaluations total
    # beta starts at 4.0 and decays to 0.5
    
    # We can estimate progress from the spread of sigma values
    # Early on, sigma is large everywhere; later, it's small in explored regions
    sigma_max = np.max(sigma)
    sigma_min = np.min(sigma)
    
    # Use sigma_max as a proxy for exploration level
    # When sigma_max is large, we're early; when small, we're late
    # Normalize to [0, 1]
    if sigma_max > 1e-6:
        progress = 1.0 - min(1.0, sigma_max / 5.0)  # 5.0 is a rough max initial sigma
    else:
        progress = 1.0
    
    # Beta decays from 4.0 to 0.5
    beta = 4.0 - 3.5 * progress
    
    # UCB: mu + beta * sigma
    ucb = mu + beta * sigma
    
    # Cost-aware scaling: divide by a convex combination of cost and a constant
    # cost is in [1, 2], so we want to penalize high-cost points
    # Use: score = ucb / (alpha * cost + (1-alpha) * c0)
    # where c0 is a reference cost (e.g., 1.0 for minimum cost)
    
    # We want to balance: lower cost is better, but we shouldn't ignore high-value points
    # alpha controls how much we weight cost vs the base score
    # Start with moderate cost sensitivity, increase as we approach budget
    
    alpha = 0.3 + 0.4 * progress  # 0.3 to 0.7
    
    c0 = 1.0  # minimum possible cost
    cost_scaled = alpha * cost + (1 - alpha) * c0
    
    # Ensure no division by zero or negative values
    cost_scaled = np.maximum(cost_scaled, 1e-6)
    
    scores = ucb / cost_scaled
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
