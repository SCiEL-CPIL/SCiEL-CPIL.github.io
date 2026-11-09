import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # f(x) ~ N(mu, sigma^2)
    # EI = E[max(f_best - f(x), 0)]
    
    # To avoid division by zero and numerical issues, handle small sigma
    # z = (f_best - mu) / sigma
    
    # We can compute EI using the standard normal CDF and PDF
    # For large negative z, EI -> 0
    # For large positive z, EI -> f_best - mu
    
    z = (f_best - mu) / sigma
    
    # Clip z to avoid overflow in exp for pdf, though scipy handles it well
    # z is typically in [-10, 10] for BO
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
