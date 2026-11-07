import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline management
_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Hypervolume Improvement (EHVI) normalized by cost.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed function value (lowest)
    - cost: array of shape (2000,), evaluation costs
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = more promising)
    """
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Deadline guard: If we are close to the 300s limit, return a simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 290:
        # Fallback: Simple EI / cost
        z = (f_best - mu) / (sigma + 1e-8)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(ei / (cost + 1e-8)), ei / (cost + 1e-8), 0.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma
    
    # Clip z for numerical stability in norm.cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: Since we are minimizing, improvement is f_best - f(x)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0) # EI is non-negative
    
    # In 1D, EHVI relative to reference point f_best is equivalent to EI
    # However, for cost-awareness, we use a cost-efficiency metric.
    # A more robust cost-aware metric is the ratio of expected improvement to cost.
    # To avoid division by very small costs dominating, we normalize cost.
    
    # Normalize cost: cost_norm = (cost - min_cost) / (max_cost - min_cost + eps)
    # But since we want lower cost to be better, we just use 1/cost or similar.
    # The standard cost-aware acquisition is EI / Cost.
    
    scores = ei / cost
    
    # Optional: Add a small exploration term if sigma is high relative to the scale
    # This helps in early stages or flat regions.
    # scale = np.max(np.abs(mu)) + 1.0
    # exploration = 0.05 * (sigma / scale)
    # scores += exploration
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
