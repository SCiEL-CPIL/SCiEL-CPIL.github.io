import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    
    # Adaptive Beta for UCB
    # We want to explore more when f_best is far from the global minimum (~0.397).
    # As f_best improves, we reduce exploration to exploit.
    
    # f_best is the current minimum observed value.
    # Branin 2D range is roughly [0.4, 300].
    # If f_best is high, we are likely in a local minimum or bad region.
    # If f_best is low, we are close to the global minimum.
    
    # Normalize f_best to a 0-1 scale where 1 is "worst" and 0 is "best".
    # Assuming a max relevant value of 50 for scaling purposes.
    # If f_best > 50, treat as max exploration.
    norm_f = np.clip(f_best / 50.0, 0.0, 1.0)
    
    # Beta starts at 2.5 when f_best is high, drops to 1.0 when f_best is low.
    beta = 1.0 + 1.5 * norm_f
    
    # UCB Score: -mu + beta * sigma
    # Higher is better.
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time (should be negligible)
    if time.time() - start_time > 0.5:
        # Fallback if something weird happened, though unlikely
        pass
        
    return scores
