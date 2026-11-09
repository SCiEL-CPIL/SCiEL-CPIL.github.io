"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) normalized by cost,
    with a small exploration bonus to aid multimodal search.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z) using erf for numerical stability and speed
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = (1 / sqrt(2 * pi)) * exp(-0.5 * z^2)
    sqrt2 = np.sqrt(2.0)
    phi_z = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    phi_z = np.maximum(phi_z, 1e-15) # Avoid underflow issues if z is huge, though exp handles it
    
    # Use erf for CDF
    cdf_z = 0.5 * (1.0 + erf(z / sqrt2))
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus to help escape local minima in Rastrigin
    # This is a soft UCB component added to EI
    exploration_bonus = 0.05 * sigma_safe
    
    total_score = ei + exploration_bonus
    
    # Normalize by cost
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = total_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
