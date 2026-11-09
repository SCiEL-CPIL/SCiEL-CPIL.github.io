"""Rewritten program using Upper Confidence Bound (UCB) for acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using the Upper Confidence Bound (UCB) strategy.
    For minimization, we want low mu and high sigma.
    Score = -mu + beta * sigma
    Higher score indicates a better candidate to evaluate next.
    """
    start_time = time.time()
    
    # Safety margin: 280 seconds out of 300 seconds allowed.
    deadline = start_time + 280.0
    
    # Ensure sigma is non-negative and not zero to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB parameter. 
    # A fixed beta of 2.0 is a standard choice for balancing exploration and exploitation.
    # This represents a structural shift from EI (improvement-based) to UCB (optimistic).
    beta = 2.0
    
    # Calculate UCB scores
    # We are minimizing, so we want to maximize (-mu + beta * sigma)
    scores = -mu + beta * sigma_safe
    
    # Check for non-finite values and replace with a very low score
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    # Deadline guard
    if time.time() > deadline:
        # Return current scores if we are close to timeout (unlikely for this size)
        pass
        
    return scores
# EVOLVE-BLOCK-END
