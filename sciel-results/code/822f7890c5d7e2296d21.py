"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We use a fixed exploration constant beta.
# In standard BO, beta is often annealed or optimized. 
# Here, we use a heuristic value that balances exploitation (low cost, low mu) 
# and exploration (high sigma, low cost).
# Since cost is in [1, 2] (1 + mean(x_scaled)), and sigma is in objective units,
# we need to scale beta appropriately.
# The GP is standardized, so sigma is in the same units as f.
# A typical beta for LCB is around 1-5 depending on the scale.
# Given the functions (Ackley, Rastrigin, Levy, Hartmann), the scale of f varies.
# However, the GP standardizes targets, so sigma is roughly O(1) relative to the range of f.
# Let's start with beta = 2.0. This encourages exploration when sigma is significant.

BETA = 2.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Parameters:
    mu: array of shape (N,) - mean predictions
    sigma: array of shape (N,) - standard deviation predictions
    f_best: float - best observed value (lowest)
    cost: array of shape (N,) - cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,) - higher is better
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle potential numerical issues
    # Avoid division by zero in cost (cost should be >= 1, but just in case)
    cost_safe = np.maximum(cost, 1e-9)
    
    # LCB for minimization: we want to minimize mu - beta * sigma
    # So the acquisition score (to be maximized) is: -(mu - beta * sigma) / cost
    # Or equivalently: (beta * sigma - mu) / cost
    
    # The cost-aware LCB score:
    # We prefer candidates with low predicted value (mu) and high uncertainty (sigma),
    # penalized by the cost of evaluation.
    
    scores = (BETA * sigma - mu) / cost_safe
    
    # Ensure finite scores
    # If sigma is 0 and mu is inf, or similar edge cases, replace with -inf or a large negative
    # But the problem states mu, sigma are from GP, so they should be finite.
    # Just in case, replace nan/inf with -inf (worst score)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
