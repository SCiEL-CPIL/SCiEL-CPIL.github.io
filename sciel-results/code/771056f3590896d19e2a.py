"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard
    # The total budget is 330s. 
    # This function is called 25 times.
    # If we are close to the deadline, we might want to fall back to a simple strategy?
    # But since this function is O(N) and N=2000, it's negligible.
    # We just return the scores.
    
    # Use UCB (Lower Confidence Bound for minimization)
    # score = mu - beta * sigma
    # Beta = 2.0 for aggressive exploration in multimodal landscape
    
    beta = 2.0
    scores = mu - beta * sigma
    
    # Ensure scores are finite
    # If sigma is 0, scores = mu.
    # If mu is inf, scores are inf.
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
