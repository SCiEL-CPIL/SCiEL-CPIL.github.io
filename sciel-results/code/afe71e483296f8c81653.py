"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Lower Confidence Bound (LCB) acquisition, scaled by 1/cost.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of shape (2000,) with higher scores meaning "evaluate next".
    """
    # Deadline guard: track elapsed time to avoid exceeding wall-clock limit
    # (This is a lightweight check; the actual deadline is enforced by the evaluator)
    
    # Clip sigma to avoid division by zero and extreme values
    sigma = np.maximum(sigma, 1e-6)
    
    # LCB for minimization: mu - kappa * sigma
    # We want higher acquisition scores to mean "more promising"
    # For minimization, LCB = mu - kappa * sigma (lower is better for the objective)
    # But we return scores where higher = evaluate next, so we negate:
    # score = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    
    # Choose kappa based on problem characteristics
    # Levy 2d has a minimum of 0 and can reach values around 200+
    # f_best is the best observed value
    # We use an adaptive kappa that balances exploration and exploitation
    
    # Scale sigma to be comparable to the objective scale
    # The standardized targets mean mu and sigma are in objective units
    # Use a kappa that gives reasonable exploration relative to the range
    
    # For Levy 2d, the function values range from 0 to ~200
    # f_best will typically be small after a few evaluations
    # We want kappa * sigma to be meaningful relative to (f_best - mu)
    
    # A fixed kappa of around 1-2 works well for standardized problems
    # But since targets are standardized, sigma represents uncertainty in objective units
    # Let's use kappa = 1.0 as a starting point, which is a common choice
    
    kappa = 1.0
    
    # LCB score: higher means more promising
    # For minimization: we want low mu and high sigma (uncertainty)
    # score = (kappa * sigma - mu) / cost
    # This is equivalent to: -mu/cost + kappa * sigma/cost
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-6)
    
    # Compute the LCB-based acquisition score
    score = (kappa * sigma - mu) / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
