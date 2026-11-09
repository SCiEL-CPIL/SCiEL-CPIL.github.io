import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) scores for candidate points.
    Uses the analytic EI formula for a Gaussian Process posterior.
    
    Parameters:
    mu : np.ndarray
        Mean of the GP posterior at candidate points, shape (N,)
    sigma : np.ndarray
        Standard deviation of the GP posterior at candidate points, shape (N,)
    f_best : float
        Current best (lowest) observed value.
        
    Returns:
    np.ndarray
        EI scores for each candidate, shape (N,). Higher scores indicate better candidates.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    # In GP, sigma should be > 0, but numerical errors can lead to 0
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization:
    # We define the improvement of a point x with value f(x) over the current best f_best as:
    # I(x) = max(0, f_best - f(x))
    # Since f(x) ~ N(mu, sigma^2), the expected improvement E[I(x)] is:
    # E[I] = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Calculate the normalized improvement
    # z = (f_best - mu) / sigma
    # Note: If mu < f_best, z is positive. If mu > f_best, z is negative.
    
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z)
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    
    # Handle potential overflow/underflow in pdf/cdf for extreme z, though norm.pdf/cdf are generally robust.
    # For very large positive z, Phi(z) -> 1, phi(z) -> 0.
    # For very large negative z, Phi(z) -> 0, phi(z) -> 0.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = (f_best - mu) * cdf_z + sigma * pdf_z
    
    ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure scores are finite
    # If there are any NaNs or Infs, replace them with -inf (worst score) to be safe,
    # but EI should be non-negative and finite for finite inputs.
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -np.inf)
    
    return ei_scores
