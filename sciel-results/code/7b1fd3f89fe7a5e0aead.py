import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # UCB for minimization: maximize -mu + beta * sigma
    # This is equivalent to minimizing mu - beta * sigma (LCB)
    beta = 1.8
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    # Deadline guard
    if time.time() - start_time > 1.0:
        # Fallback: Greedy (minimize mu)
        scores = -mu
        
    return scores
