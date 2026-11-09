"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # We want to maximize EI, so return EI directly as the score.
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    z = (f_best - mu) / sigma_safe
    
    # EI formula
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle edge cases: where sigma is very small, EI should be 0 if mu < f_best else 0
    # Actually, when sigma -> 0, EI -> max(f_best - mu, 0)
    # But the formula above handles this naturally as z -> inf or -inf
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI should be non-negative; clamp any tiny negative values from numerical errors
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
