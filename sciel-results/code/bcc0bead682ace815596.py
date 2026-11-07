import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Probability of Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    scores = norm.cdf(z)
    
    # Deadline guard (not strictly necessary for this fast op, but requested)
    elapsed = time.time() - start_time
    if elapsed > 300:
        # Return best found (which is just the scores)
        return scores
        
    return scores
