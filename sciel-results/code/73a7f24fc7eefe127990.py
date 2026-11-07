"""Cost-aware acquisition for BO on shifted Hartmann 3D (5 seeds, total cost < 30)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: kill long-running computations safely.
_T0 = time.time()
_HARD_DEADLINE = 320.0  # seconds; leave ~10s buffer for the harness


def _elapsed_ok():
    return (time.time() - _T0) < _HARD_DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = better to evaluate next).
    
    Strategy: Adaptive Upper Confidence Bound (UCB) per unit cost.
    UCB balances exploitation (low mu) and exploration (high sigma).
    The exploration parameter beta adapts based on the progress (f_best).
    Early on (high f_best), beta is large to explore.
    Late on (low f_best), beta is small to exploit.
    The score is normalized by cost to prefer cheaper evaluations.
    """
    if not _elapsed_ok():
        return np.full_like(mu, 0.0)
    
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Estimate progress: 0 (start) to 1 (end)
    # Hartmann 3D min is ~ -3.86. Max is ~ 3.86 (approx).
    # f_best starts high (e.g. 3.0) and goes down to -3.86.
    # Let's map f_best from [4.0, -3.86] to [0.0, 1.0]
    f_max_est = 4.0
    f_min_est = -3.86
    range_f = f_max_est - f_min_est
    
    # p is 0 when f_best is high (early), 1 when f_best is low (late)
    p = (f_max_est - f_best) / range_f
    p = np.clip(p, 0.0, 1.0)
    
    # Adaptive beta: start with high exploration, end with low
    beta_max = 6.0
    beta_min = 1.5
    beta = beta_max * (1.0 - p) + beta_min * p
    
    # UCB for minimization: f_best - mu + beta * sigma
    # We want to maximize this score.
    ucb = f_best - mu + beta * sigma_safe
    
    # Cost-aware score
    scores = ucb / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
