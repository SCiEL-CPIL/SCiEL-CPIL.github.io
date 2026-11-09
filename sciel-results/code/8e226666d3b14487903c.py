import numpy as np
import time

# Global time tracker for deadline guard
_start_time = None

def _init_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def acquisition(mu, sigma, f_best):
    _init_time()
    # Deadline guard: The acquisition function is fast, but we check just in case.
    # If we are extremely close to the 330s limit, we might want to bail out or ensure no hangs.
    # However, the operations below are O(N) vectorized and very fast.
    
    # Upper Confidence Bound (UCB) for minimization.
    # We want to maximize the score, where higher score means "evaluate next".
    # For minimization, we prefer low mu (exploitation) and high sigma (exploration).
    # Score = -mu + beta * sigma
    beta = 1.5
    
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
