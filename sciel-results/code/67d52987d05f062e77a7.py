"""Rewritten program using Expected Improvement for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 320.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    mu: array of shape (2000,) - posterior mean in objective units
    sigma: array of shape (2000,) - posterior standard deviation in objective units
    f_best: float - best (lowest) observed value so far
    
    Returns: array of shape (2000,) - scores, higher means more promising
    """
    # Check deadline - if we're close to timeout, return a simple exploitation score
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        # Fallback: pure exploitation (negative mean)
        return -mu

    # Handle cases where sigma is very small to avoid numerical issues
    sigma_safe = np.where(sigma < 1e-8, 1e-8, sigma)
    
    # Calculate the improvement parameter
    # For minimization, improvement is f_best - f(x)
    # We want to maximize the expected improvement
    improvement = f_best - mu
    
    # Standard normal CDF and PDF
    Z = improvement / sigma_safe
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # Expected Improvement formula for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: if improvement is negative (mu > f_best), EI is small/zero
    # We add a small exploration term to ensure we don't get stuck if sigma is tiny everywhere
    # But EI naturally handles this: if sigma is small and mu > f_best, EI is ~0.
    # If sigma is large, EI is large.
    
    ei = improvement * Phi + sigma_safe * phi
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Optional: Add a small amount of exploration noise if we suspect stagnation?
    # For now, pure EI is the standard.
    
    return ei
# EVOLVE-BLOCK-END
