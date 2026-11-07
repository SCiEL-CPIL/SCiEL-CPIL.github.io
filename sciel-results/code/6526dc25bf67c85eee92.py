import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    This is a refinement of the LCB family that dynamically balances exploration
    and exploitation based on the current best value f_best.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate the z-score: how many standard deviations f_best is above the mean
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive, indicating potential improvement.
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    
    # Use scipy.stats.norm for numerical stability
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    scores = (f_best - mu) * Phi + sigma * phi
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # EI is always non-negative. Higher is better.
    return scores
