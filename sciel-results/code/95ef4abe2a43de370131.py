"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill
_TIMEOUT_MARGIN = 25.0  # return at least 25s before deadline

def _check_timeout():
    return (time.time() - _START_TIME) > (_DEADLINE - _TIMEOUT_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Lower Confidence Bound (LCB).
    
    Instead of Expected Improvement (EI), which integrates the tail of the distribution,
    this uses an optimistic bound (mu - kappa * sigma) to drive exploration.
    This structural change can be more robust to GP uncertainty calibration issues
    and may help escape local minima in multi-modal functions like Rastrigin by
    prioritizing regions with high potential for improvement under uncertainty.
    
    The exploration coefficient kappa decays as the total cost increases,
    shifting the strategy from exploration to exploitation.
    """
    if _check_timeout():
        # Fallback: simple cost-aware exploitation
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: non-positive sigma
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    
    # Adaptive exploration coefficient
    # Start with high exploration (kappa=2.5) and decay to low exploitation (kappa=0.5)
    # Cost ranges from ~5 (initial) to 30.
    # We map cost to a decay factor.
    # exp(-cost / 10) goes from exp(-0.5)~0.6 to exp(-3)~0.05
    # Let's use a linear decay for predictability: kappa = 2.5 - 0.1 * cost
    # At cost=5, kappa=2.0. At cost=30, kappa=0.5.
    kappa = 2.5 - 0.1 * np.mean(cost)
    kappa = np.clip(kappa, 0.5, 3.0)
    
    # Lower Confidence Bound (for minimization)
    lcb = mu - kappa * sigma_safe
    
    # Potential improvement: how much better could the LCB be than f_best?
    # We only care if the LCB is better than f_best.
    improvement = np.maximum(0.0, f_best - lcb)
    
    # Cost-aware score
    cost_safe = np.where(cost > 1e-8, cost, 1e-8)
    score = improvement / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
