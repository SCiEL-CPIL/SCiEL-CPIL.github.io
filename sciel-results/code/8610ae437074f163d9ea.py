"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Upper Confidence Bound (UCB) for minimization.
    # We want to minimize the function, so we look for low mean (mu) and high uncertainty (sigma).
    # The score to maximize is -mu + beta * sigma.
    # We divide by cost to prefer cheaper evaluations.
    
    # Ensure cost is not zero to avoid division errors
    cost_safe = np.maximum(cost, 1e-9)
    
    # Beta controls exploration vs exploitation.
    # A value around 1.0-2.0 is typical for UCB in low-dimensional BO.
    beta = 1.5
    
    # Calculate UCB score
    ucb_score = -mu + beta * sigma
    
    # Adjust for cost
    score = ucb_score / cost_safe
    
    return score
# EVOLVE-BLOCK-END
