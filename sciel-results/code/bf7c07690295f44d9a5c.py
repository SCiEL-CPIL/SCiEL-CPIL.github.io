"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Strategy: Cost-aware LCB (Lower Confidence Bound) with a dynamic
    exploration coefficient that shifts from high to low as budget depletes.
    The cost penalty is applied globally to discourage expensive evaluations.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Estimate progress: use f_best relative to expected range
    # Powell 4D on [-4,5]^4 has min 0, typical initial values ~10-20
    f_range = max(abs(f_best), 1.0) + 1.0
    progress = 1.0 - (f_best / (f_range * 0.5))
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic beta: high exploration early, low exploitation late
    # Start with beta ~3.0, end with beta ~0.5
    beta = 3.0 - 2.5 * progress
    
    # Cost-aware LCB: minimize mu - beta * sigma, penalize high cost
    # The cost penalty should be stronger when cost is high
    # Use a scaled penalty: penalty = alpha * cost / (1 + cost)
    # This is bounded and emphasizes cost differences in the mid-range
    alpha = 0.5 + 0.5 * progress  # stronger penalty as we progress
    
    # Compute the raw LCB score (lower is better for minimization)
    lcb = mu - beta * sigma_safe
    
    # Cost penalty: higher cost means lower score (since we want higher = evaluate next)
    # We subtract a penalty proportional to cost
    cost_penalty = alpha * np.log1p(cost_safe)
    
    # Final score: higher is better
    # We want to maximize: -lcb - cost_penalty = -mu + beta*sigma - cost_penalty
    score = -lcb - cost_penalty
    score = mu * (-1) + beta * sigma_safe - cost_penalty
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
