import numpy as np

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero or negative variance issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score.
    # The score represents the "optimistic" minimum value we might find.
    # score = -mean + beta * std
    beta = 1.0
    
    scores = -mu + beta * sigma_safe
    
    # Clean up numerical noise
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
