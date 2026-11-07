import numpy as np
import time

# Store the start time of the module load or first call to track elapsed time
# However, the evaluator calls acquisition repeatedly.
# We should track time relative to the start of the *entire* run if possible, 
# but we don't have access to the start time of the evaluator.
# The prompt says "return the best result found with a safety margin".
# Since acquisition is stateless and fast, this is mostly defensive.
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 10.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Args:
        mu: Mean of GP posterior, shape (N,)
        sigma: Std dev of GP posterior, shape (N,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher is better.
    """
    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # If we are out of time, return a neutral score or best effort.
        # Returning zeros is safe as it implies no preference.
        # However, we must return finite scores.
        return np.zeros_like(mu)

    # Generate random samples from the posterior
    # z ~ N(0, 1)
    z = np.random.standard_normal(size=mu.shape)
    
    # Sampled objective values
    y = mu + sigma * z
    
    # Ensure y is finite
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Cost-aware score: -y / cost
    # We want to minimize y, so lower y is better.
    # We want to minimize cost, so lower cost is better.
    # Score = -y / cost.
    # If y is small (good), -y is large (positive if y negative, small negative if y positive).
    # If cost is small, division increases the magnitude.
    # Let's verify the direction.
    # We want to MAXIMIZE the score.
    # Candidate A: y=0.1, cost=1. Score = -0.1
    # Candidate B: y=0.1, cost=2. Score = -0.05
    # B is preferred (lower cost). Correct.
    # Candidate C: y=0.01, cost=2. Score = -0.005
    # C is preferred over B (lower y). Correct.
    # Candidate D: y=1.0, cost=1. Score = -1.0
    # A is preferred over D (lower y). Correct.
    
    # Add a small epsilon to cost to avoid division by zero, though cost >= 1
    cost_safe = np.maximum(cost, 1e-6)
    
    scores = -y / cost_safe
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
