"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: track elapsed time and return best result found with safety margin.
    # The function is called 2000 times? No, it's called once per iteration with 2000 candidates.
    # The loop runs while cost < 30. 
    # We need to ensure this function itself doesn't hang.
    # It's a vectorized operation, so it should be fast.
    
    # UCB (Upper Confidence Bound) formulation
    # Score = (mu + beta * sigma) / cost
    # This balances exploration (sigma) and exploitation (mu) while penalizing cost.
    
    # Beta parameter for UCB. 
    # The knowledge base suggests beta=2 worked well (score 1.034).
    beta = 2.0
    
    # Ensure sigma is non-negative and handle potential numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate UCB score
    ucb = mu + beta * sigma
    
    # Normalize by cost
    # cost is in [1, 2]
    scores = ucb / cost
    
    # Return finite scores
    # If any score is not finite, replace with a very low value
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
