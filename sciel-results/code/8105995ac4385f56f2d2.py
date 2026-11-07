"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB (Upper Confidence Bound) for minimization.
    
    Args:
        mu: Mean of the GP prediction, shape (2000,)
        sigma: Standard deviation of the GP prediction, shape (2000,)
        f_best: Best observed value so far
        cost: Cost of each candidate, shape (2000,)
        
    Returns:
        Scores for each candidate, higher is better.
    """
    # Deadline guard: If we are close to the 330s limit, return a safe fallback
    # to ensure the program doesn't hang or crash due to timeout in the evaluator's loop.
    # We use a 300s limit as a safety margin.
    if time.time() - _START_TIME > 300:
        # Fallback: Just return negative cost (prefer cheaper points) to stop expensive searches
        return -cost / np.max(cost)

    # UCB for minimization:
    # We want to find low values.
    # Score should be high for low mu and high sigma (uncertainty).
    # UCB_score = -mu + beta * sigma
    # We divide by cost to account for cost-awareness.
    
    # Exploration parameter beta.
    # A value around 1.5-2.5 is common.
    beta = 2.0
    
    # Calculate UCB scores
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    ucb_scores = -mu + beta * sigma
    
    # Normalize by cost
    # Cost is >= 1, so division is safe.
    scores = ucb_scores / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
