"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Args:
        mu: array of shape (n,) - posterior means
        sigma: array of shape (n,) - posterior standard deviations
        f_best: float - best observed value so far
    
    Returns:
        array of shape (n,) - acquisition scores (higher is better for selection)
    """
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate Z-score for Expected Improvement
    # We are minimizing, so improvement is f_best - f(x)
    z = (f_best - mu) / sigma
    
    # Standard Normal PDF: phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    phi_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    
    # Standard Normal CDF: Phi(z)
    Phi_z = ndtr(z)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # If f_best - mu is negative (mu > f_best), EI should be 0 or close to 0.
    # The formula handles this naturally as Phi(z) becomes small and (f_best - mu) is negative,
    # but strictly speaking EI is max(0, ...). However, the continuous formula 
    # (f_best - mu) * Phi(z) + sigma * phi(z) is the standard integral result 
    # for E[max(f_best - Y, 0)] where Y ~ N(mu, sigma^2).
    
    score = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite and non-negative (though EI is naturally non-negative)
    score = np.maximum(score, 0.0)
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
