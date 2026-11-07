import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) acquisition function with a small threshold.
    Scores candidates by the probability that they will be significantly better 
    than the current best.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Define a small threshold for 'significant' improvement
    # This helps avoid converging prematurely to local minima
    xi = 1e-3
    
    # Compute z-score for P(f(x) < f_best - xi)
    z = (f_best - xi - mu) / sigma_safe
    
    # Compute Probability of Improvement
    scores = norm.cdf(z)
    
    # Sanitize scores to ensure they are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
