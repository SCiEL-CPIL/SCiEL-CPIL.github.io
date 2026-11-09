import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB for minimization
    # Beta decreases as f_best gets closer to 0 (the optimum)
    # Max possible f is 25 (at x=5 or x=-5)
    # We map f_best to a beta value.
    
    # Clip f_best to [0, 25] to be safe
    f_best_clipped = np.clip(f_best, 0.0, 25.0)
    
    # Beta ranges from 0.1 (exploit) to 3.0 (explore)
    beta_max = 3.0
    beta_min = 0.1
    
    # Linear interpolation
    # If f_best is 25, beta is beta_max
    # If f_best is 0, beta is beta_min
    beta = beta_min + (beta_max - beta_min) * (f_best_clipped / 25.0)
    
    # UCB score for minimization: -mu + beta * sigma
    scores = -mu + beta * sigma
    
    return scores
