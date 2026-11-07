"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
# 330s wall clock limit, keep 10s margin for safety
_DEADLINE = _START_TIME + 320

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) with robust numerical handling.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE - 5:
        # Fallback: prefer cheap points if we are running out of time
        cost_safe = np.maximum(cost, 1e-9)
        return 1.0 / cost_safe

    # Sanitize inputs to prevent numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Calculate z-score for Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to a reasonable range to ensure numerical stability
    z = np.clip(z, -10.0, 10.0)
    
    # Compute EI components
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Add a small exploration bonus to encourage exploring high-uncertainty regions
    # This helps escape local optima in the Rastrigin function
    exploration_bonus = 0.05 * sigma_safe
    
    # Total acquisition value
    acquisition_value = ei + exploration_bonus
    
    # Cost-aware: EI per unit cost
    scores = acquisition_value / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
