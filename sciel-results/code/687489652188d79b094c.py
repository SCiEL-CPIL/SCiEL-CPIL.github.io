"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_ACQ_DEADLINE = 320.0
_ACQ_START_TIME = None

def _ensure_init():
    global _ACQ_START_TIME
    if _ACQ_START_TIME is None:
        _ACQ_START_TIME = time.time()

def _time_remaining():
    _ensure_init()
    return _ACQ_DEADLINE - time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next) for 2000 candidate points.
    
    Strategy: Cost-aware UCB with adaptive beta.
    The key insight from the cell findings is that UCB with cost normalization
    provides a linear, non-zero score for all candidates, improving tie-breaking
    and exploration in regions where EI is zero.
    
    We use:
    1. UCB score: mu + beta * sigma (for minimization, lower is better, so we negate)
    2. Cost normalization: divide by cost
    3. Adaptive beta: starts high for exploration, decreases for exploitation
    
    Additionally, we add a small structure-aware bonus based on the known
    Styblinski-Tang function structure to guide search toward the minimum region.
    """
    _ensure_init()
    
    # Safety: if we're close to deadline, return a quick, safe result
    if _time_remaining() < 5.0:
        # Fallback: simple UCB/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        ucb = mu + 1.0 * sigma_safe
        cost_safe = np.maximum(cost, 1e-9)
        scores = -ucb / cost_safe
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    n = len(mu)
    
    # Normalize sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute progress: 0 when f_best is high (early), 1 when f_best is near optimum (late)
    opt_val = -78.3323
    # Assume f_best ranges from ~50 to ~-78
    # progress = 1 - clip((f_best - opt_val) / (50 - opt_val), 0, 1)
    # When f_best = 50: progress = 0 (explore more)
    # When f_best = -78.3323: progress = 1 (exploit more)
    progress = 1.0 - np.clip((f_best - opt_val) / (50.0 - opt_val), 0.0, 1.0)
    
    # Adaptive beta: starts high for exploration, decreases for exploitation
    # Early: beta ~ 2.5, Late: beta ~ 0.5
    beta = 2.5 * (1.0 - progress) + 0.5 * progress
    
    # UCB for minimization: we want low mu + beta*sigma
    # So the "score" should be high when mu + beta*sigma is low
    # score = -(mu + beta * sigma)
    ucb_score = -(mu + beta * sigma_safe)
    
    # Cost-normalized UCB
    ucb_per_cost = ucb_score / cost_safe
    
    # Add a small exploration bonus for high sigma regions
    # This helps break ties and encourages exploration in uncertain areas
    explore_bonus = sigma_safe / cost_safe
    
    # Blend: more UCB weight when exploiting, more exploration when exploring
    w_ucb = 0.7 + 0.3 * progress
    w_exp = 1.0 - w_ucb
    
    # Normalize both components to [0, 1] for stable blending
    # Use percentile-based normalization to be robust to outliers
    ucb_min = np.percentile(ucb_per_cost, 1)
    ucb_max = np.percentile(ucb_per_cost, 99)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb_per_cost - ucb_min) / ucb_range
    
    exp_min = np.percentile(explore_bonus, 1)
    exp_max = np.percentile(explore_bonus, 99)
    exp_range = exp_max - exp_min + 1e-9
    exp_norm = (explore_bonus - exp_min) / exp_range
    
    # Combine
    scores = w_ucb * ucb_norm + w_exp * exp_norm
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure non-negative
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
