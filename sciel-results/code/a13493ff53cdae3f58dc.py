"""Acquisition function for Bayesian optimization of Beale 2D using Student-t Process EI."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import t

# Track start time for deadline enforcement
_START_TIME = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Student-t Process Expected Improvement.
    
    Parameters:
    - mu: array of shape (2000,) with posterior means in objective units
    - sigma: array of shape (2000,) with posterior standard deviations in objective units
    - f_best: lowest observed value so far
    
    Returns:
    - array of shape (2000,) with scores; higher means more promising
    """
    # Check deadline - if we're close to timeout, return a simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Fallback: return negative mu (pure exploitation) to finish quickly
        return -mu

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Student-t Process parameters
    # Using nu=3 degrees of freedom for heavier tails
    nu = 3.0
    
    # Scale for the t-distribution
    # If Y ~ t_nu(mu, sigma^2/nu), then Y = mu + s * T, where T ~ t_nu(0,1)
    # Variance of T is nu/(nu-2). Variance of Y is s^2 * nu/(nu-2).
    # We want Var(Y) = sigma^2.
    # s^2 * nu/(nu-2) = sigma^2 => s = sigma * sqrt((nu-2)/nu)
    # However, the GP posterior gives sigma as the standard deviation of the Gaussian.
    # In STP, the predictive variance is often modeled differently.
    # A common approximation is to map the GP sigma to the t-distribution scale.
    # Let's use s = sigma / sqrt(nu) as a simple heuristic often used in literature 
    # or s = sigma * sqrt((nu-2)/nu) to match variance.
    # Matching variance: s = sigma * sqrt((3-2)/3) = sigma / sqrt(3).
    s = sigma_safe / np.sqrt(nu)
    
    # Z-score for the t-distribution
    Z = (f_best - mu) / s
    
    # Compute CDF and PDF of the standard t-distribution
    # scipy.stats.t.cdf and pdf are vectorized
    F_t = t.cdf(Z, nu)
    f_t = t.pdf(Z, nu)
    
    # Expected Improvement for Student-t
    # EI = s * [ Z * F_t(Z) + f_t(Z) * (nu / (nu-1)) * (1 + Z^2 / nu) ]
    
    # Calculate the term
    term1 = Z * F_t
    term2 = f_t * (nu / (nu - 1.0)) * (1.0 + (Z ** 2) / nu)
    
    ei = s * (term1 + term2)
    
    # Handle potential numerical issues (e.g., if Z is very large, f_t might be 0, which is fine)
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # The score is the Expected Improvement. Higher is better.
    return ei

# EVOLVE-BLOCK-END
