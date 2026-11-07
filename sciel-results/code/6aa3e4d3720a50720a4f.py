import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes Expected Improvement (EI) acquisition scores for minimization.
    
    Parameters:
    mu: array of shape (N,), mean of the GP posterior.
    sigma: array of shape (N,), standard deviation of the GP posterior.
    f_best: float, the best (lowest) observed function value so far.
    
    Returns:
    scores: array of shape (N,), EI values. Higher is better.
    """
    start_time = time.time()
    limit = 0.5  # 500ms limit per call, should be well under this
    
    # Ensure sigma is non-negative and not too small to avoid division issues
    # If sigma is 0, the point is known exactly.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement z = (f_best - mu) / sigma
    # For minimization, we want f(x) < f_best.
    # Improvement = max(f_best - f(x), 0)
    # E[Improvement] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Compute CDF and PDF of the standard normal distribution
    # norm.cdf(z) and norm.pdf(z) are vectorized
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement
    # Note: (f_best - mu) can be negative if mu > f_best.
    # In that case, z is negative, cdf_z is small, pdf_z is small.
    # The term (f_best - mu) * cdf_z will be negative, but pdf_z term is positive.
    # EI should be non-negative.
    # Actually, if mu > f_best, the expected improvement is small but positive due to the tail.
    # The formula naturally handles this.
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Numerical safety: EI should be non-negative.
    # Due to floating point errors, it might be slightly negative.
    ei = np.maximum(ei, 0.0)
    
    # Check time
    if time.time() - start_time > limit:
        # Fallback: If EI calculation took too long (unlikely), 
        # fall back to a simple exploitation strategy: pick lowest mu.
        # This is a safety net.
        scores = -mu
    else:
        scores = ei
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
