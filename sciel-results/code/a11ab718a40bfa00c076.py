"""Rewritten acquisition function using Cost-Aware EI with Exploration Bonus."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard (330s wall-clock; keep ~10s margin)
_DEADLINE = time.time() + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Combines Expected Improvement (EI) with a pure exploration term (sigma),
    normalized by the evaluation cost.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Stable EI calculation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term: Use sigma directly.
    # We weight it to balance exploitation (EI) and exploration (sigma).
    # A weight of 1.0 suggests that uncertainty is valued equally to expected improvement.
    # Given the small budget, exploration is crucial.
    exploration_weight = 1.0
    exploration_term = exploration_weight * sigma_safe
    
    # Combined score before cost adjustment
    raw_score = ei + exploration_term
    
    # Cost-aware adjustment
    scores = raw_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time - if we're getting close to the limit, return immediately
    if time.time() > _DEADLINE:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
