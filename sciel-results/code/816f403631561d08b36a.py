"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    normalized by evaluation cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed objective value
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure numerical stability for sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is standard normal CDF, phi is standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Handle numerical issues where z is very large or small
    # For z > 0, EI is positive. For z < 0, EI is also positive (potential improvement).
    # The formula holds generally.
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical noise might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: Expected Improvement per unit cost
    # Cost is in [1, 2], so division is stable.
    # We use a small epsilon in the denominator to prevent division by zero if cost is 0 (though it shouldn't be).
    cost_safe = np.maximum(cost, eps)
    
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
