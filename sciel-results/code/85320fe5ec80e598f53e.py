import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes Expected Improvement (EI) scores for minimization.
    
    Parameters:
    mu: array-like, shape (n_candidates,), posterior mean in objective units
    sigma: array-like, shape (n_candidates,), posterior std dev in objective units
    f_best: float, best (lowest) observed value in objective units
    
    Returns:
    scores: array-like, shape (n_candidates,), EI values (higher is better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle potential numerical issues with sigma
    # If sigma is extremely small, the distribution is a point mass.
    # We use a small epsilon to prevent division by zero.
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # Note: We are minimizing, so improvement is f_best - f(x).
    z = (f_best - mu) / sigma_safe
    
    # Compute PDF and CDF of standard normal
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where z = (f_best - mu) / sigma
    improvement_term = (f_best - mu) * cdf
    uncertainty_term = sigma * pdf
    
    scores = improvement_term + uncertainty_term
    
    # Ensure scores are finite
    # If mu is NaN or Inf, scores might be invalid.
    # Replace non-finite values with -inf so they are never selected (since we maximize)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
