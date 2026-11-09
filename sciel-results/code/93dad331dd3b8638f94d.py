"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def _ei_fallback(mu, sigma, f_best):
    """Fallback Expected Improvement score."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    from scipy.stats import norm
    return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Gaussian Process Thompson Sampling (GPTS).
    Draws a sample from the posterior GP and returns negative sample values
    as scores (higher score = lower sampled value = more promising).
    
    Fallback to EI if Cholesky decomposition fails or time is exceeded.
    """
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return _ei_fallback(mu, sigma, f_best)
    
    n = len(mu)
    
    # Try Thompson Sampling
    try:
        # Build covariance matrix: K_ij = sigma_i * sigma_j * (1 - delta_ij * noise)
        # For sampling, we use the covariance structure of the GP
        # The posterior covariance at test points given training data is:
        # K_tt - K_tf K_ff^{-1} K_ft
        # But we only have mu and sigma, not the full GP internals.
        # 
        # Approximation: Assume the posterior covariance is proportional to sigma_i * sigma_j
        # This is a simplification, but captures the uncertainty structure.
        #
        # For Thompson Sampling, we need to sample from N(mu, C) where C is the posterior covariance.
        # Since we don't have the full covariance matrix, we approximate:
        # C_ij ≈ sigma_i * sigma_j * rho_ij
        # where rho_ij is the correlation between points i and j.
        #
        # Without coordinates, we cannot compute rho_ij exactly.
        # However, we can use a simpler approach:
        # Sample f_i ~ N(mu_i, sigma_i^2) independently.
        # This is equivalent to EI with kappa=1 in some sense, but simpler.
        
        # Independent sampling (diagonal covariance approximation)
        rng = np.random.default_rng()
        sample = mu + sigma * rng.standard_normal(n)
        
        # Score: higher is better, so negate the sample
        score = -sample
        
        # Ensure finite
        score = np.where(np.isfinite(score), score, 0.0)
        
        return score
        
    except Exception:
        # Fallback to EI
        return _ei_fallback(mu, sigma, f_best)
# EVOLVE-BLOCK-END
