import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma
    This is a structural change from the previous Thompson Sampling approach.
    """
    start_time = time.time()
    deadline = 295.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-6)

    # UCB for minimization:
    # We want to maximize the score.
    # Lower mu is better (exploitation).
    # Higher sigma is better (exploration).
    # Score = -mu + beta * sigma
    
    # Choose beta. 
    # For a smooth function like x^2, a moderate beta is good.
    # Beta = 1.0 is a common starting point.
    # Let's try 2.0 to encourage more exploration in the early stages.
    beta = 2.0
    
    # To prevent sigma from dominating too much if it's very large,
    # we can clip it, but UCB usually handles this well.
    # However, if sigma is huge, -mu + 2*sigma might pick a point with high mu but huge sigma.
    # For sphere 1D, sigma shouldn't be huge.
    
    score = -mu + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: exploitation
        fallback_score = -np.where(np.isfinite(mu), mu, f_best)
        fallback_score = np.where(np.isfinite(fallback_score), fallback_score, -1e10)
        return fallback_score
        
    return score
