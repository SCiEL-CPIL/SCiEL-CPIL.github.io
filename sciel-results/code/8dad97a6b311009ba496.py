import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB acquisition function for minimization
    # We want to maximize the score, which should favor low mu and high sigma.
    # Score = -mu + beta * sigma
    beta = 2.0
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
