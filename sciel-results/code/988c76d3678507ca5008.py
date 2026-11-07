import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # We maximize E[max(0, f_best - f(x))]
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Z-score for the improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure non-negative scores (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    return ei
