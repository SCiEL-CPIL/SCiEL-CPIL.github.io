"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_time_limit = 290  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) for minimization.
    
    Args:
        mu: Array of mean predictions (2000,)
        sigma: Array of standard deviation predictions (2000,)
        f_best: Scalar, the best observed value so far.
        cost: Array of costs for each candidate (2000,)
        
    Returns:
        Array of acquisition scores (2000,)
    """
    # Check time budget - if we're close to the limit, return simple scores quickly
    if time.time() - _start_time > _time_limit:
        return -mu / cost
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is standard normal CDF and phi is standard normal PDF
    
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)  # Clip to avoid numerical issues
    
    # Use scipy for normal CDF and PDF if available, otherwise approximate
    try:
        from scipy.stats import norm
        Phi = norm.cdf(z_clipped)
        phi = norm.pdf(z_clipped)
    except ImportError:
        # Fallback: use numpy approximation
        # Phi(z) approximation
        Phi = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * z_clipped))
        # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
        phi = np.exp(-0.5 * z_clipped**2) / np.sqrt(2.0 * np.pi)
    
    # EI for minimization: positive when mu < f_best (improvement expected)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Cost-aware: divide by cost to prefer cheaper points
    # Use a power of cost to balance exploration vs exploitation
    cost_power = 1.0
    scores = ei / (cost ** cost_power)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
