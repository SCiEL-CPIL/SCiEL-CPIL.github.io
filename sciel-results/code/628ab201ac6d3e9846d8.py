# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track time since first call to stay within budget.
_t_start = None

def acquisition(mu, sigma, f_best, cost):
    global _t_start
    if _t_start is None:
        _t_start = time.time()
    
    # Safety: if we're running out of time, return a simple but valid score
    elapsed = time.time() - _t_start
    if elapsed > 310.0:
        # Return a valid finite array immediately
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        pi = norm.cdf(z)
        return pi / (cost + eps)
    
    # Cost-aware Probability of Improvement (PI) acquisition.
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This directly targets regions where improvement is likely,
    # which is more robust to Rastrigin's multi-modal structure than EI.
    # Dividing by cost creates a trade-off: high probability AND low cost.
    
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    pi = norm.cdf(z)
    
    # Cost-aware score: probability of improvement per unit cost
    score = pi / (cost + eps)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
