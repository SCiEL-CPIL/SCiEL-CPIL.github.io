import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300 seconds as per task constraint

def acquisition(mu, sigma, f_best, cost):
    # Check if we're close to the deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a simple fallback if we're running out of time
        # Just return EI without cost penalty to be fast
        sigma_safe = np.clip(sigma, 1e-9, None)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.clip(sigma, 1e-9, None)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but numerical errors might make it tiny negative)
    ei = np.maximum(ei, 0.0)
    
    # Apply cost-aware penalty
    # Cost ranges from 1 to 2 (1 + mean(x_scaled) where x_scaled in [0,1])
    # We want to prefer lower-cost evaluations
    # Using a multiplicative penalty that decays with cost
    # Normalize cost: cost_norm = (cost - 1) / 1.0 = cost - 1, ranges [0, 1]
    cost_norm = cost - 1.0
    
    # Use a softer penalty to allow exploration
    # Try a milder penalty: exp(-0.5 * cost_norm)
    # This gives penalty range from 1.0 (cost=1) to ~0.607 (cost=2)
    
    penalty = np.exp(-0.5 * cost_norm)
    
    scores = ei * penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
