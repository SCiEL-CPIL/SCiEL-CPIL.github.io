import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    # Higher EI means more expected improvement, so higher score is better.
    
    # Handle zero or negative sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    return ei
