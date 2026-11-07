"""Rewritten cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock budget is ~330s. We track per-call time
# but more importantly keep each call fast (vectorized, O(n) or O(n log n)).
# Since the loop runs until cost < 30, and each eval costs ~1 to 2,
# that's at most ~15-20 evaluations. Each call processes 2000 candidates.
# The bottleneck is the norm.cdf/pdf calls, which are O(n).

# We'll use a cost-aware UCB (Upper Confidence Bound) acquisition function.
# UCB = -mu + kappa * sigma, where kappa controls exploration.
# For cost-awareness, we divide by cost to prefer cheaper evaluations.
# Key insight: cost = 1 + mean(x_scaled), so points near x=[5,5] (scaled [1,1]) 
# are more expensive. We should prefer cheap points that have good UCB.

# To be safe against numerical issues, we clip values and handle edge cases.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition: (-mu + kappa*sigma) / cost, with safeguards.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP std dev predictions (positive)
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive kappa: start with high exploration, decrease as we get closer to optimum
    # Use f_best to gauge progress: if f_best is close to 0, we're near optimum, reduce exploration
    # Three hump camel minimum is 0, so f_best approaching 0 means we're close
    # Scale kappa based on how far f_best is from the known minimum (0)
    # When f_best is large (far from min), use high kappa; when f_best ~ 0, use low kappa
    # Use a smooth transition: kappa = base_kappa * exp(-f_best / scale)
    # But f_best can be negative for three hump camel? No, min is 0, so f_best >= 0
    # Actually three hump camel can have negative values, but global min is 0
    # Let's use: if f_best < 0.1, we're close, reduce exploration
    # Use a simple adaptive scheme:
    # kappa = 0.5 when f_best > 1, decreasing to 0.1 when f_best < 0.1
    # Linear interpolation in log space for smoothness
    f_best_safe = max(f_best, 1e-10)  # avoid log(0)
    # Map f_best from [0.01, 10] to kappa from [0.1, 0.5]
    # Use log scale: log(f_best) from log(0.01) to log(10)
    log_fbest = np.log(f_best_safe)
    log_fbest_min = np.log(0.01)
    log_fbest_max = np.log(10.0)
    # Clip to range
    log_fbest_clipped = np.clip(log_fbest, log_fbest_min, log_fbest_max)
    # Normalize to [0, 1]
    t = (log_fbest_clipped - log_fbest_min) / (log_fbest_max - log_fbest_min)
    # t=0 when f_best=0.01 (close to min), t=1 when f_best=10 (far from min)
    # kappa = 0.1 + t * (0.5 - 0.1) = 0.1 + 0.4*t
    kappa = 0.1 + 0.4 * t
    
    # UCB for minimization: we want to minimize, so we prefer low mu and high sigma
    # UCB = -mu + kappa * sigma (higher is better for selection)
    ucb = -mu + kappa * sigma_safe
    
    # Cost-aware: divide by cost to prefer cheap evaluations
    # Add small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    score = ucb / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Normalize scores to [0, 1] range for stability (optional but helps with scaling)
    score_min = np.min(score)
    score_max = np.max(score)
    if score_max > score_min + 1e-12:
        score = (score - score_min) / (score_max - score_min)
    
    return score
# EVOLVE-BLOCK-END
