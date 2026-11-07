"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses Expected Improvement (EI) for minimization.
    EI is a robust choice that balances exploration and exploitation
    based on the posterior distribution provided by the GP.
    
    For a Gaussian posterior N(mu, sigma^2), the Expected Improvement
    over the current best f_best is:
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    Phi is the standard normal CDF, phi is the standard normal PDF.
    
    This approach adapts naturally: where sigma is high, EI encourages
    exploration; where mu is low, EI encourages exploitation.
    """
    # Handle numerical stability for sigma close to 0
    # If sigma is very small, the distribution is nearly deterministic.
    # EI approaches max(f_best - mu, 0)
    
    # We add a small epsilon to sigma to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # term1 = (f_best - mu) * Phi(z)
    # term2 = sigma * phi(z)
    term1 = (f_best - mu) * norm.cdf(z)
    term2 = sigma_safe * norm.pdf(z)
    
    score = term1 + term2
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
