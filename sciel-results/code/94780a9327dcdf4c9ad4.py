"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Computes the acquisition score for each candidate.
    Uses cost-aware Lower Confidence Bound (LCB) with dynamic beta.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 290s safety margin before 300s limit
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    eps = 1e-9
    
    # Dynamic beta: scale exploration based on uncertainty
    # Use a fixed beta that balances exploration and exploitation
    # For Rastrigin, which is multimodal, we want aggressive exploration
    # beta is chosen to be large enough to explore but not too large to be unstable
    beta = 2.0
    
    # LCB score: (f_best - mu + beta * sigma) / cost
    # We want to minimize, so we maximize the "optimistic gap"
    # If mu is low and sigma is high, we get a high score
    # Dividing by cost makes it cost-aware
    
    numerator = (f_best - mu) + beta * sigma
    scores = numerator / (cost + eps)
    
    # Check deadline
    if time.time() > deadline:
        # Return best effort so far
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
