import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute the improvement term
    improvement = f_best - mu
    
    # Standardize
    z = improvement / sigma
    
    # Use scipy for accurate CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = improvement * cdf + sigma * pdf
    
    # Ensure finite output
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Higher EI means better candidate
    scores = ei
    
    return scores
