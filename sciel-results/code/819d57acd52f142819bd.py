"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a bounded exploration bonus.
    
    Calculates the expected improvement over f_best for each candidate,
    adds a bounded exploration term, and normalizes by evaluation cost.
    Higher scores indicate better candidates.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed value so far (lowest since we're minimizing).
    cost : array-like, shape (2000,)
        Cost of evaluating each point, shape (2000,).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate Z score: (f_best - mu) / sigma
    # If mu < f_best, Z is positive.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Where Phi is CDF and phi is PDF of standard normal distribution.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Bounded exploration bonus: sigma / (sigma + 0.1)
    # This keeps exploration from dominating early while still guiding
    # search toward uncertain regions. The bounded form prevents the
    # exploration weight from overwhelming EI when sigma is large.
    exploration = sigma_safe / (sigma_safe + 0.1)
    
    # Combine EI with bounded exploration, then divide by cost
    scores = (ei + exploration) / cost_safe

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
