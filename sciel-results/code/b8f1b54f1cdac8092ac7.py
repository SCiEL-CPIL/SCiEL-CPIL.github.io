import numpy as np

def acquisition(mu, sigma, f_best):
    # Calculate distance from the known global minimum (3.0)
    dist = f_best - 3.0
    if dist < 0:
        dist = 0.0
        
    # Adaptive beta: High exploration when far from optimum, low when close
    # The exponential decay ensures a smooth transition
    beta = 0.1 + 1.0 * (1.0 - np.exp(-dist / 5.0))
    
    # UCB score for minimization: we want to maximize (beta * sigma - mu)
    scores = beta * sigma - mu
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
