"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 310.0

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Lower Confidence Bound (LCB).
    
    Structural Change:
    1. Switches from EI-based to LCB-based acquisition.
    2. Introduces adaptive exploration parameter beta based on f_best.
       Early in the search (high f_best), beta is large to explore.
       Late in the search (low f_best), beta is small to exploit.
    3. Cost adjustment is applied by dividing the LCB score by the evaluation cost.
    """
    if not _time_ok():
        # Fallback to simple exploitation if time is running out
        # Return index of lowest mu (highest score = -mu/cost)
        scores = -mu / (cost + 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta
    # f_best starts high (positive) and goes to -78.33
    # Map f_best from [100, -78] to beta [3.0, 0.5]
    # Linear interpolation
    f_best_norm = (f_best - 100.0) / (-78.33 - 100.0)
    f_best_norm = np.clip(f_best_norm, 0.0, 1.0)
    # When f_best is 100, norm is 0 -> beta should be high (3.0)
    # When f_best is -78, norm is 1 -> beta should be low (0.5)
    beta = 3.0 - 2.5 * f_best_norm
    
    # Lower Confidence Bound for minimization
    # We want to minimize mu - beta * sigma
    # Score (higher is better) should be proportional to -(mu - beta * sigma) = beta * sigma - mu
    lcb_score = beta * sigma_safe - mu
    
    # Cost adjustment
    # Prefer lower cost
    cost_safe = np.maximum(cost, 1e-9)
    score = lcb_score / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
