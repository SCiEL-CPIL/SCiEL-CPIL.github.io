import numpy as np
import time

# Global state for time tracking if needed, but usually local is fine per call.
# The evaluator kills the whole process after 330s.
# We don't know the start time of the whole process easily inside the function unless we store it.
# However, the function is called repeatedly.
# Let's assume the "deadline" applies to the total execution.
# Since we can't easily access the global start time without side-effects or global variables,
# and the computation is very fast, a simple check might be overkill but requested.
# I will use a global start time.

_start_time = time.time()
_TIMEOUT_LIMIT = 320.0 # Safety margin before 330s

def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: Array of shape (N,) with mean predictions.
        sigma: Array of shape (N,) with standard deviation predictions.
        f_best: Scalar, the best observed value so far.
        
    Returns:
        Array of shape (N,) with scores. Higher scores are preferred.
    """
    # Check time limit
    current_time = time.time()
    elapsed = current_time - _start_time
    
    # If we are dangerously close to the timeout, return a safe fallback.
    # The fallback should be fast and deterministic.
    # A simple fallback: maximize -mu (greedy) to ensure we finish.
    if elapsed > _TIMEOUT_LIMIT:
        return -mu
    
    # LCB: Minimize mu - beta * sigma
    # We want to maximize the score, so score = -(mu - beta * sigma)
    beta = 2.0
    
    # Ensure sigma is non-negative (it should be, but just in case)
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -(mu - beta * sigma)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
