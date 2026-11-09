import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # Closed form: (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Ensure sigma is not zero to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Calculate EI
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure all scores are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
