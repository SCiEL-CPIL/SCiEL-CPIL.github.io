import numpy as np

def acquisition(mu, sigma, f_best):
    # Ensure sigma is non-negative and avoid division by zero if we were doing EI, 
    # but for UCB we just multiply.
    sigma = np.maximum(sigma, 1e-9)
    
    # Deterministic UCB (Lower Confidence Bound for minimization)
    # We want to maximize the score, which corresponds to minimizing (mu - kappa * sigma)
    # Score = -(mu - kappa * sigma) = -mu + kappa * sigma
    
    # High kappa for aggressive exploration in multimodal landscape
    kappa = 5.0
    
    scores = -mu + kappa * sigma
    
    return scores
