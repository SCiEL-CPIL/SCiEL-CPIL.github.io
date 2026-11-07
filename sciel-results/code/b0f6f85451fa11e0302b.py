import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # We want to maximize the score, which corresponds to minimizing the Lower Confidence Bound (LCB).
    # LCB = mu - beta * sigma
    # Score = -LCB = -mu + beta * sigma
    
    # A beta of 2.0 is a common default that balances exploration and exploitation well for 2D problems.
    beta = 2.0
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0)
    
    score = -mu + beta * sigma
    
    # Return finite scores
    return score
