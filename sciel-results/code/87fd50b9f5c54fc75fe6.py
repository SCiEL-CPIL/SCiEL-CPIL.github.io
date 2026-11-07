"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 290  # Safety margin for the 300s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Variance-Gated Expected Improvement (VEI).
    
    This approach computes the standard Expected Improvement (EI) for minimization
    and then scales it by a factor based on the relative uncertainty (sigma).
    This structurally penalizes exploitation in regions where the GP is already
    confident (low sigma), encouraging the search to explore high-uncertainty
    regions, which is critical for escaping local minima in multimodal functions
    like Branin.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Check deadline to ensure we return valid scores quickly if time is running out
    if time.time() > _DEADLINE:
        # Fallback: simple UCB-like score
        return -mu + 2.0 * sigma

    # Ensure sigma is strictly positive to avoid division by zero or log(0)
    # A small epsilon prevents numerical issues
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate standard Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Variance Gating
    # We scale the EI by (sigma / sigma_max)^gamma
    # This forces the acquisition to prioritize points with higher uncertainty.
    # gamma = 1.5 provides a strong but not absolute penalty for low uncertainty.
    sigma_max = np.max(sigma_safe)
    if sigma_max > eps:
        # Relative uncertainty
        rel_sigma = sigma_safe / sigma_max
        # Gate factor
        # As sigma -> 0, gate -> 0. As sigma -> sigma_max, gate -> 1.
        # This effectively zeros out EI in well-explored areas.
        gate = np.power(rel_sigma, 1.5)
    else:
        gate = np.ones_like(sigma_safe)
        
    scores = ei * gate
    
    # Return finite scores
    return scores
# EVOLVE-BLOCK-END
