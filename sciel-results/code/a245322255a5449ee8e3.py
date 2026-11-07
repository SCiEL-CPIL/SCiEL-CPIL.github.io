import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI) with jitter.
    
    Parameters:
    mu (array-like): Mean predictions of the GP, shape (2000,).
    sigma (array-like): Standard deviation predictions of the GP, shape (2000,).
    f_best (float): The best observed function value so far.
    
    Returns:
    array: Scores for each candidate, higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    # and handle any non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Probability of Improvement for minimization
    # PI(x) = Phi( (f_best - mu(x)) / sigma(x) )
    # We add a small jitter to sigma to regularize the score and prevent
    # extreme sensitivity to very small sigma values or to dampen
    # exploration of regions with very high uncertainty.
    epsilon = 1e-4
    z = (f_best - mu) / (sigma + epsilon)
    
    # Compute the CDF of the standard normal distribution
    scores = norm.cdf(z)
    
    # Ensure outputs are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
