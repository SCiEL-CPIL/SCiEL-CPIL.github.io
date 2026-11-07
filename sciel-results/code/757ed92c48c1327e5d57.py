"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with sublinear cost discount.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Current best observed value.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure safety margins for sigma and cost
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Handle non-finite values
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        # Fallback: return uniform scores if data is corrupt
        return np.ones_like(mu)
    
    # Calculate Expected Improvement (EI) for minimization
    # imp = f_best - mu
    # z = imp / sigma
    # EI = imp * Phi(z) + sigma * phi(z)
    
    imp = f_best - mu
    z = imp / sigma
    
    # Clip z to avoid overflow/underflow in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: Sublinear cost discount
    # Using 1/sqrt(cost) instead of 1/cost to reduce bias towards cheap regions
    # while still penalizing high cost.
    cost_discount = 1.0 / np.sqrt(cost)
    
    scores = ei * cost_discount
    
    # Final sanity check
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
