import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) with a small exploration bonus.
    """
    # Compute z-score for each candidate
    # For minimization, improvement = f_best - f(x)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Handle zero sigma to avoid division by zero
    safe_sigma = np.where(sigma > 1e-10, sigma, 1e-10)
    z = (f_best - mu) / safe_sigma
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima by favoring high-uncertainty regions
    # The bonus is scaled by the maximum sigma to keep it in a reasonable range
    max_sigma = np.max(sigma) if len(sigma) > 0 else 1.0
    if max_sigma > 1e-10:
        exploration_bonus = 0.1 * sigma / max_sigma
    else:
        exploration_bonus = np.zeros_like(sigma)
    
    # Higher EI means more likely improvement, so we want to maximize EI
    # Return EI + exploration bonus (higher is better)
    scores = ei + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
