import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    UCB Acquisition Function for Minimization.
    Score = -mu + beta * sigma
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback to greedy to ensure fast return
        return -mu
    
    # UCB parameters
    # Beta controls exploration. 
    # 1.0 is conservative, 2.0 is aggressive.
    # Given few iterations (30), higher exploration is often better.
    beta = 2.0
    
    # Calculate scores
    # mu and sigma are arrays of shape (2000,)
    scores = -mu + beta * sigma
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
