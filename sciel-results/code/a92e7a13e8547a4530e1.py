import numpy as np
from scipy.stats import norm

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Args:
        mu: Array of posterior means (shape 2000,).
        sigma: Array of posterior standard deviations (shape 2000,).
        f_best: The best (lowest) observed value so far.
        
    Returns:
        Array of scores (shape 2000,). Higher scores indicate more promising candidates.
    """
    # Ensure sigma is non-negative and handle zeros to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # For points where sigma is effectively zero, EI is just the improvement if mu < f_best
    # The formula above handles this well if sigma is small, but let's be explicit for stability
    # If sigma is very small, z is large (if mu < f_best) or very negative (if mu > f_best).
    # If mu < f_best, z is large positive, Phi(z) ~ 1, phi(z) ~ 0, EI ~ f_best - mu.
    # If mu > f_best, z is large negative, Phi(z) ~ 0, phi(z) ~ 0, EI ~ 0.
    
    # Clip negative EI to 0 just in case of numerical errors (shouldn't happen with correct formula)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
