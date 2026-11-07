import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB
    # Base beta 2.0, max beta 4.0
    # Scale factor based on how far f_best is from 0
    # If f_best is large, we need more exploration
    beta = 2.0 + 2.0 * min(1.0, max(0.0, f_best) / 5.0)
    
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores
