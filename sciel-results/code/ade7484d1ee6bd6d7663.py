# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Approach: Cost-Weighted Upper Confidence Bound (UCB) for minimization.
    # This approach explores regions with high uncertainty (sigma) while exploiting
    # regions with low predicted values (mu). It is distinct from Expected Improvement
    # which focuses on the expected magnitude of improvement over f_best.
    
    # Safety checks for division
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Exploration parameter. 
    # A value of 2.0 provides a balance between exploitation and exploration.
    beta = 2.0
    
    # Calculate UCB score for minimization.
    # We want to maximize the score, so we negate mu (lower is better) 
    # and add a term for uncertainty.
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-awareness: divide by cost to prefer cheaper evaluations
    score = ucb_score / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
