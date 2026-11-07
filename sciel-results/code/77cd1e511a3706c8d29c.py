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
    # Use f_best to gauge progress.
    # f_best large (far from min) -> High exploration (High Kappa).
    # f_best small (close to min) -> Low exploration (Low Kappa).
    # Use log-scale interpolation for smoothness.
    f_best_safe = max(f_best, 1e-10)
    log_fbest = np.log(f_best_safe)
    # Map f_best from [0.01, 10] to t in [0, 1]
    # t=0 for f_best=0.01 (good), t=1 for f_best=10 (bad)
    log_fbest_min = np.log(0.01)
    log_fbest_max = np.log(10.0)
    log_fbest_clipped = np.clip(log_fbest, log_fbest_min, log_fbest_max)
    t = (log_fbest_clipped - log_fbest_min) / (log_fbest_max - log_fbest_min)
    
    # Increase exploration range: kappa from 0.2 (exploit) to 1.0 (explore)
    kappa = 0.2 + 0.8 * t
    
    # UCB for minimization: -mu + kappa * sigma
    ucb = -mu + kappa * sigma_safe
    
    # Cost-aware: Use a softer penalty than division.
    # Cost is in [1, 2].
    # score = ucb / (1 + 0.5 * (cost - 1))
    # This penalizes expensive points but less aggressively than ucb / cost.
    # This helps if the optimum is in a moderately expensive region (like center).
    cost_penalty = 1.0 + 0.5 * (np.maximum(cost, 1.0) - 1.0)
    score = ucb / cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Return scores directly. Normalization is not required for ranking and can be skipped
    # to save computation and avoid potential numerical issues with identical scores.
    return score
# EVOLVE-BLOCK-END
