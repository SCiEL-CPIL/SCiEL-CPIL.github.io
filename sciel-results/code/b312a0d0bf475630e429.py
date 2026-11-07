import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Knowledge Gradient (KG) approximation for minimization.
    KG estimates the expected reduction in the global minimum value for each candidate.
    We approximate this by computing the expected improvement over f_best,
    which serves as a proxy for how likely a point is to yield a new global best.
    """
    # Compute expected improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    
    # Handle cases where sigma is very small to avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-10)
    
    # Standardized distance from f_best
    z = (f_best - mu) / safe_sigma
    
    # Expected improvement formula
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Clip negative values to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Return EI as the score (higher is better)
    scores = ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
