"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level state to track elapsed time across calls
_start_time = None


def _get_elapsed():
    """Get elapsed time since the first call, initializing if needed."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a structural change: instead of simple EI/cost, this uses
    a combination of:
    1. Expected Improvement (EI) for exploitation
    2. UCB (Upper Confidence Bound) for exploration
    3. Cost-aware weighting that accounts for the budget structure
    
    The key insight is that with a fixed budget of 30 and initial cost of ~5,
    we have limited evaluations. We need to balance exploration and exploitation
    more aggressively than simple EI/cost.
    """
    # Deadline guard: if we're running long, return a safe default
    elapsed = _get_elapsed()
    if elapsed > 280:  # Safety margin before 300s limit
        # Return a simple, fast score: prefer low cost, low mu
        cost_safe = np.maximum(cost, 1e-10)
        scores = -mu / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB for exploration: mu - beta * sigma (for minimization)
    # Use a beta that adapts based on how much uncertainty there is
    # When sigma is large relative to the range, explore more
    beta = 1.5  # Moderate exploration
    
    # For minimization, UCB score = -mu + beta * sigma (we want to maximize this)
    ucb = -mu + beta * sigma_safe
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-10)
    
    # Combine EI and UCB with cost-aware weighting
    # Weight EI more when we have good information (low sigma), 
    # weight UCB more when uncertain (high sigma)
    # Normalize sigma to [0, 1] range for weighting
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    sigma_norm = sigma_safe / (sigma_max + 1e-10)
    
    # EI weight: higher when sigma is low (confident)
    ei_weight = 1.0 - 0.5 * sigma_norm
    # UCB weight: higher when sigma is high (uncertain)
    ucb_weight = 0.5 * sigma_norm + 0.3
    
    # Compute combined score
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(np.abs(ucb)) if np.max(np.abs(ucb)) > 0 else 1.0
    
    ei_norm = ei / (ei_max + 1e-10)
    ucb_norm = ucb / (ucb_max + 1e-10)
    
    # Combined acquisition score before cost normalization
    combined = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Cost-aware normalization: divide by cost
    scores = combined / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
