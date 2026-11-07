"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit
_SAFETY_MARGIN = 15.0  # seconds to leave before the limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Returns 2000 finite scores (higher = evaluate next).
    
    Strategy: Blended Cost-Aware EI and UCB (Kriging Believer style).
    - EI captures the expected gain over current best.
    - UCB (mu - beta*sigma) captures the optimistic potential (believer) for new minima.
    - Both are normalized and blended to balance exploitation and exploration.
    - Scaled by cost to favor cheaper evaluations.
    """
    # Deadline guard: if we're getting close to the limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: Simple cost-aware EI without complex normalization
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        f_best = float(f_best)
        
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        
        delta = f_best - mu
        z = delta / sigma_safe
        z = np.clip(z, -10, 10)
        
        ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        
        scores = ei / cost_safe
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # 1. Compute Expected Improvement (EI) for minimization
    delta = f_best - mu
    z = delta / sigma_safe
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_cost = ei / cost_safe

    # 2. Compute UCB (Kriging Believer / Optimistic Lower Bound) for minimization
    # We want to explore where mu is low AND sigma is high.
    # ucb_min = mu - beta * sigma
    # Higher score is better, so we use -ucb_min
    beta = 2.0
    ucb_min = mu - beta * sigma_safe
    ucb_score = -ucb_min / cost_safe

    # 3. Normalize both components to [0, 1] range for stable blending
    # Add small epsilon to avoid division by zero if range is 0
    eps = 1e-9
    
    ei_min = np.min(ei_cost)
    ei_max = np.max(ei_cost)
    ei_range = ei_max - ei_min + eps
    ei_norm = (ei_cost - ei_min) / ei_range
    
    ucb_min_val = np.min(ucb_score)
    ucb_max_val = np.max(ucb_score)
    ucb_range = ucb_max_val - ucb_min_val + eps
    ucb_norm = (ucb_score - ucb_min_val) / ucb_range

    # 4. Blend: 70% cost-aware EI, 30% cost-aware UCB
    # This balances the expected gain (EI) with the potential for new minima (UCB)
    weight_ei = 0.7
    weight_ucb = 0.3

    scores = weight_ei * ei_norm + weight_ucb * ucb_norm

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final safety check for any remaining NaNs/Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
# EVOLVE-BLOCK-END
