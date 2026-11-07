import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard: ensure we return within time limits
    # The evaluator has a 300s limit per evaluation, but we should be fast anyway
    
    # Use Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-10)
    improvement = f_best - mu
    
    # Avoid division by zero issues
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # Compute Phi(z) and phi(z)
    # Using scipy for accuracy if available, otherwise approximate
    try:
        from scipy.stats import norm
        phi = norm.pdf(z)
        Phi = norm.cdf(z)
    except ImportError:
        # Approximate using erf
        from math import erf, sqrt
        Phi = 0.5 * (1 + erf(z / sqrt(2)))
        phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    
    # Expected Improvement for minimization
    # When improvement > 0 (mu < f_best), EI should be positive
    # EI = improvement * Phi(z) + sigma * phi(z)
    scores = improvement * Phi + sigma_safe * phi
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
