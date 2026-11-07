import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    
    # LCB parameter
    # Beta controls exploration. Higher beta -> more exploration.
    # For Beale function with small budget, moderate exploration is key.
    beta = 3.0
    
    # Compute LCB score
    # We want to minimize (mu - beta * sigma)
    # Score (higher is better) = -(mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Safety margin check (unlikely to trigger for vectorized ops on 2000 items)
    elapsed = time.time() - start_time
    if elapsed > 10.0:
        # If it took too long, return a conservative score?
        # But this is just arithmetic, so it won't happen.
        pass
        
    return scores
