"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted UCB for minimization.
    mu, sigma, cost: shape (2000,)
    f_best: scalar
    Returns: shape (2000,) finite scores, higher = evaluate next.
    """
    start_time = time.time()
    deadline = 300.0
    
    # Ensure sigma is positive to avoid division issues, though UCB doesn't divide by sigma
    sigma = np.maximum(sigma, 1e-12)
    
    # Cost-Adjusted UCB for minimization
    # We want to minimize mu - beta * sigma.
    # So the "desirability" (score to maximize) is -(mu - beta * sigma) = -mu + beta * sigma.
    # We normalize by cost to account for evaluation cost.
    
    # Beta controls exploration. 
    # A value of 2.0 is a standard starting point for UCB.
    # Since the function is multimodal (Styblinski-Tang), we might need higher exploration.
    # Let's try beta = 2.0.
    
    beta = 2.0
    
    # Calculate UCB desirability
    ucb_desirability = -mu + beta * sigma
    
    # Normalize by cost
    # cost is in [1, 2]
    score = ucb_desirability / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
