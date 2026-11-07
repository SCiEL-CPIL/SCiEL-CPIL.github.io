import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function.
    Returns scores for 2000 candidates. Higher scores indicate higher probability
    of improvement over f_best.
    """
    # Avoid division by zero in sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    scores = norm.cdf(z)
    
    return scores
