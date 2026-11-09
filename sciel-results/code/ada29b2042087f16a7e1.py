"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global variable to track start time for deadline guard
_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def acquisition(mu, sigma, f_best):
    # Set start time on first call
    _get_start_time()
    
    # Deadline guard: If we are very close to the 330s limit, 
    # we might want to switch to a faster, more exploitative strategy 
    # or just ensure we return quickly. 
    # However, since this function is vectorized and fast, 
    # we just proceed. The guard is a safety net.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # If we are near the deadline, prioritize exploitation to avoid 
        # wasting time on exploration that won't be evaluated anyway.
        # Return -mu (lower mu is better, so higher score for lower mu)
        # But the function expects "higher meaning evaluate next".
        # So we want high scores for low mu.
        # score = -mu
        return -mu

    # Upper Confidence Bound (UCB)
    # score = mu + beta * sigma
    beta = 2.0
    scores = mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
