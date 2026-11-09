import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    Returns scores where higher is better.
    score = beta * sigma - mu
    """
    # Ensure sigma is non-negative and handle potential numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Beta controls exploration vs exploitation
    # Higher beta -> more exploration
    beta = 1.5
    
    # For minimization, we want low mu and high sigma.
    # We return a score to be maximized.
    # LCB minimizes (mu - beta * sigma).
    # Maximizing (beta * sigma - mu) is equivalent.
    scores = beta * sigma - mu
    
    return scores
