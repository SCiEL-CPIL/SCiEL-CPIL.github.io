"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # 290 seconds limit per evaluation, keep safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a cost penalty and exploration bonus.
    
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
        # Fallback: return a simple cost-aware score
        return 1.0 / (cost + 1e-9)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Cost-aware: normalize EI by cost
    # Use a soft cost penalty to balance exploitation and cost efficiency
    cost_penalty = cost + 1e-9
    
    # Primary score: EI per unit cost
    score_ei = ei / cost_penalty
    
    # Exploration bonus: encourage exploring uncertain regions
    # But weighted to not overwhelm the EI signal
    exploration_weight = 0.1
    exploration_score = sigma_safe / cost_penalty
    
    # Combine: primarily EI, with a small exploration bonus
    scores = score_ei + exploration_weight * exploration_score
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip to avoid extreme values
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores
# EVOLVE-BLOCK-END
