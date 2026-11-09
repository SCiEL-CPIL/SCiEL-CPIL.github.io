"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Global state to track number of evaluations for adaptive exploration
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: if we're running long, just return something safe
    # (though with 2000 points and simple math, this shouldn't be an issue)
    
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is CDF and phi is PDF of standard normal
    # Higher EI means more improvement expected
    
    # Handle case where sigma is very small to avoid division by zero
    # When sigma is near zero, EI should be max(f_best - mu, 0)
    
    # Add small epsilon to sigma to prevent division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute the improvement term
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: for minimization, we want to find points where mu < f_best
    # The standard EI formula is for maximization, so we adapt it
    
    # For minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x)^2)
    # This equals:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    
    # Compute Phi and phi
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Compute EI
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Clip negative values to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
