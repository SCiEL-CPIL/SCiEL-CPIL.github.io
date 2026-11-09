import numpy as np
import time

_START_TIME = time.time()
_SAFETY_MARGIN = 325.0
_CALL_COUNT = 0

def acquisition(mu, sigma, f_best):
    global _CALL_COUNT
    
    # Deadline Guard
    if time.time() > _SAFETY_MARGIN:
        return -mu

    # Increment call count
    _CALL_COUNT += 1
    
    # Total BO steps expected: 30 total evals - 5 initial = 25 steps
    # We assume this function is called once per step.
    total_steps = 25
    current_step = min(_CALL_COUNT, total_steps)
    
    # Adaptive Beta
    # Start with high exploration, end with high exploitation
    beta_max = 4.0
    beta_min = 0.5
    
    # Linear decay
    progress = current_step / total_steps
    beta = beta_max - (beta_max - beta_min) * progress
    
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
    
    return scores
