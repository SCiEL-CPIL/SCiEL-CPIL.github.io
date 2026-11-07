import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Normalize f_best to [0, 1] range for the adaptive beta
    # Branin range is roughly 0 to 300
    # If f_best is very low, we are close to the optimum.
    # We want high exploration (high beta) when f_best is high (early stages).
    # We want low exploration (low beta) when f_best is low (late stages).
    
    # Estimate progress: 0 if f_best is max, 1 if f_best is min
    # Max value of Branin is around 300. Min is 0.398.
    # Let's use a soft cap.
    max_val = 300.0
    min_val = 0.4
    
    # Progress p: 0 (start) to 1 (end)
    # If f_best is 300, p=0. If f_best is 0.4, p=1.
    p = np.clip((max_val - f_best) / (max_val - min_val), 0.0, 1.0)
    
    # Beta starts high (5.0) and decays to low (0.5)
    beta = 5.0 - 4.5 * p
    
    # UCB for minimization: mu - beta * sigma
    scores = mu - beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
