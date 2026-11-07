"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
# Safety margin: The limit is 330s. We aim to finish within 320s to be safe.
_DEADLINE = _START_TIME + 320.0
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    
    # Explicit deadline guard
    if time.time() > _DEADLINE:
        # Fallback: Greedy selection based on mean
        return -mu
        
    _call_count += 1
    
    # We expect about 25 calls (30 total evals - 5 initial).
    # Use a UCB (Upper Confidence Bound) acquisition function.
    # Score = -mu + beta * sigma
    # Higher score means higher priority for evaluation.
    # This balances exploitation (low mu) and exploration (high sigma).
    
    # Dynamic beta: Start with high exploration, decay to lower exploration.
    # Total calls ~ 25.
    # beta = 4.0 - 0.12 * call_count
    # At call 0: beta = 4.0
    # At call 24: beta = 4.0 - 2.88 = 1.12
    # This ensures strong exploration early on to find global minima,
    # and more exploitation later to refine the best found.
    
    beta = 4.0 - 0.12 * _call_count
    if beta < 0.5:
        beta = 0.5
        
    # Ensure sigma is non-negative
    sigma = np.abs(sigma)
    
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
