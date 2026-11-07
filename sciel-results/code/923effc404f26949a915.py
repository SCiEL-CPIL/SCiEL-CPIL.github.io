import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI formula:
    # (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Calculate terms
    term1 = (f_best - mu) * norm.cdf(z)
    term2 = sigma_safe * norm.pdf(z)
    
    ei = term1 + term2
    
    # Handle cases where sigma is extremely small (deterministic case)
    # If sigma is very small, the function is effectively at mu.
    # EI should be max(f_best - mu, 0).
    # The formula above might suffer from numerical issues if z is very large positive/negative.
    # norm.cdf(z) -> 1 for large z, norm.pdf(z) -> 0.
    # If z is large positive (mu << f_best), EI ~ f_best - mu.
    # If z is large negative (mu >> f_best), EI ~ 0.
    
    # To be safe against numerical underflow/overflow in norm.pdf for very large |z|:
    # If z > 10, pdf is ~0, cdf is ~1. EI ~ f_best - mu.
    # If z < -10, pdf is ~0, cdf is ~0. EI ~ 0.
    
    # Let's refine for numerical stability
    # For very small sigma, just use the max improvement
    mask_small_sigma = sigma < 1e-6
    if np.any(mask_small_sigma):
        ei[mask_small_sigma] = np.maximum(f_best - mu[mask_small_sigma], 0)
        
    return ei
