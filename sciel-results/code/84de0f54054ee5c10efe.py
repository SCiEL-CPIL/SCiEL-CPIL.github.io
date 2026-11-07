import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize this score.
    # Low mu is good (exploitation).
    # High sigma is good (exploration).
    # Score = -mu + beta * sigma
    
    # Tune beta for 3D Hartmann with ~30 evaluations.
    # A value between 2 and 5 is typical. 3.0 is a safe, strong choice.
    beta = 3.0
    
    # Ensure sigma is non-negative to avoid issues, though it should be from GP
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma
    
    return scores
