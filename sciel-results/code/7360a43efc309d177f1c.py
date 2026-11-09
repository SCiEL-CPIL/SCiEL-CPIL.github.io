import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with an exploration bonus.
    """
    # Ensure sigma is positive for division
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration bonus (0.1 * sigma)
    # This encourages exploration of high-uncertainty regions
    score = ei + 0.1 * sigma
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
