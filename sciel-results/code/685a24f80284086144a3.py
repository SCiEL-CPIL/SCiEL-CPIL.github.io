import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline management
_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a soft cost penalty and exploration bonus.
    
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
        scores = ei / (cost + 1e-8)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma
    
    # Clip z for numerical stability in norm.cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term: Uncertainty (Sigma)
    # This helps escape local optima and explore regions with high variance
    kappa = 0.5
    exploration = kappa * sigma
    
    # Combined value: Exploitation (EI) + Exploration (Sigma)
    combined = ei + exploration
    
    # Soft Cost Penalty
    # Instead of linear division (cost ** 1), use a power < 1 to soften the bias
    # towards cheap regions, allowing the GP to guide search to potentially 
    # better but slightly more expensive areas.
    alpha = 0.8
    cost_penalty = np.power(cost, alpha)
    
    scores = combined / cost_penalty
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
