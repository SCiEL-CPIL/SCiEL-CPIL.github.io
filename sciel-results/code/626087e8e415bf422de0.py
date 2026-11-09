import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # Equivalent to UCB for maximization of -f
    # Score = beta * sigma - mu
    # Higher score indicates a more promising candidate
    
    # Choose beta parameter for exploration/exploitation trade-off
    # For small sample sizes (30 evals), a higher beta encourages exploration
    beta = 2.0
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0)
    
    # Calculate score
    # We want to minimize f, so we want low mu and high sigma (uncertainty)
    # The term -mu rewards low mean values
    # The term beta * sigma rewards high uncertainty
    score = beta * sigma_safe - mu
    
    return score
