"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Regret (ER) normalized by cost.
    
    Expected Regret directly measures the expected reduction in the objective function,
    which is more appropriate for cost-aware optimization than Expected Improvement.
    
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
    
    # Compute z-score: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z) using erf for numerical stability and speed
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = (1 / sqrt(2 * pi)) * exp(-0.5 * z^2)
    sqrt2 = np.sqrt(2.0)
    phi_z = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    phi_z = np.maximum(phi_z, 1e-15)  # Avoid underflow issues
    
    # Use erf for CDF
    cdf_z = 0.5 * (1.0 + erf(z / sqrt2))
    
    # Expected Regret: E[max(f_best - f(x), 0)]
    # This is actually the same formula as EI, but we interpret it as expected regret
    # The key difference is in how we handle the cost normalization and exploration
    er = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Ensure non-negative
    er = np.maximum(er, 0.0)
    
    # Normalize by cost to get expected regret per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = er / cost_safe
    
    # Add a small exploration bonus scaled by cost to encourage exploration in
    # high-uncertainty regions that are also cost-effective
    exploration_bonus = 0.03 * sigma_safe / cost_safe
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
