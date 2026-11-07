import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    This strategy uses a linear combination of the posterior mean and standard deviation,
    normalized by the current best value to be scale-invariant.
    
    Parameters:
    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    scores: array of shape (2000,), acquisition values (higher is better)
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Handle edge case where f_best is 0 or very small
    # Beale minimum is 0. If f_best is 0, we are done, but we must return scores.
    # If f_best is 0, mu and sigma should be around 0.
    # To avoid division by zero, we use a small epsilon.
    f_best_safe = np.maximum(f_best, 1e-9)
    
    # Normalize mu and sigma by f_best to make the acquisition function scale-invariant.
    # This allows us to use a fixed beta parameter.
    mu_norm = mu / f_best_safe
    sigma_norm = sigma_safe / f_best_safe
    
    # UCB for minimization:
    # We want to maximize the score.
    # A point is good if it has a low mean (exploit) or high uncertainty (explore).
    # Score = -mu_norm + beta * sigma_norm
    
    # Beta controls the exploration/exploitation trade-off.
    # Beta = 1.0 is a standard starting point for normalized UCB.
    beta = 1.0
    
    scores = -mu_norm + beta * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf (shouldn't happen), return zeros
    if np.all(scores == -np.inf):
        return np.zeros_like(mu)
        
    return scores
