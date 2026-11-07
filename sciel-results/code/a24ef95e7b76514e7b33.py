import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    For minimization, we want to select points with low mean (mu) and high uncertainty (sigma).
    The score is defined as: Score = beta * sigma - mu.
    The evaluator selects the point with the highest score.
    
    A beta of 2.5 provides strong exploration, which is beneficial for the
    multi-modal Styblinski-Tang function within the limited 30-evaluation budget.
    """
    beta = 2.5
    
    # Ensure sigma is strictly positive to avoid any numerical edge cases, 
    # although multiplication is safe, this is a standard precaution.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate UCB scores
    scores = beta * sigma_safe - mu
    
    return scores
