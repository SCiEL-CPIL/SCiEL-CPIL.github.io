import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    EI(x) = E[max(0, f_best - f(x))]
    
    For a Gaussian posterior N(mu, sigma^2):
    EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    where Z = (f_best - mu) / sigma
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Ensure sigma is positive and avoid division by zero
    eps = 1e-12
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Calculate Z-score
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    
    # Clip Z to a reasonable range to avoid numerical issues in norm.cdf/pdf
    # For Z > 8, Phi(Z) ~ 1 and phi(Z) ~ 0
    # For Z < -8, Phi(Z) ~ 0 and phi(Z) ~ 0
    Z_clipped = np.clip(Z, -10, 10)
    
    # Calculate EI components
    # (f_best - mu) is the potential improvement
    improvement = f_best - mu
    
    # Phi(Z) and phi(Z)
    cdf = norm.cdf(Z_clipped)
    pdf = norm.pdf(Z_clipped)
    
    # EI = improvement * Phi(Z) + sigma * phi(Z)
    # Note: if Z is very negative, improvement is negative, but Phi(Z) is ~0.
    # We should ensure we don't have negative EI due to floating point errors.
    EI = improvement * cdf + sigma_safe * pdf
    
    # EI should be non-negative. Clip to 0 to handle numerical noise.
    EI = np.maximum(EI, 0.0)
    
    # Ensure finiteness
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    return EI
