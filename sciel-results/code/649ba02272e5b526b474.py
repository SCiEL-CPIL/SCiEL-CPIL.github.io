import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Ensure sigma is positive to avoid division by zero errors in z-score calculation
    # Although sigma is a standard deviation, numerical noise might make it 0 or negative.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi is CDF, phi is PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle cases where sigma is effectively 0
    # If sigma is very small, the distribution is a spike.
    # If mu < f_best, the improvement is f_best - mu.
    # If mu >= f_best, the improvement is 0.
    # The formula above handles this well as z -> +/- infinity, but let's be explicit for robustness if needed.
    # Actually, the analytic formula is stable enough for 1e-8 floor.
    
    return ei
