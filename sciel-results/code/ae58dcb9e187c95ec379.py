"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for the deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 330s limit

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with adaptive cost penalty.
    
    Strategy:
    1. Compute the probability of improvement P(f(x) < f_best) using the GP predictive mean/std.
    2. Apply a cost penalty that is more aggressive for higher costs, specifically 
       using an exponential decay to concentrate search near the center (low cost) 
       where the shifted Levy minimum is likely to reside.
    3. The score is the probability of improvement multiplied by the cost penalty.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement (POI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Adaptive Cost Penalty
    # The cost ranges from 1.0 (center) to 2.0 (corners) roughly.
    # We want to heavily penalize high cost points to keep the search in the "cheap" central region,
    # especially as we might be running out of budget (though we don't have explicit budget left here,
    # the exponential form is generally more effective for Levy functions where the minimum is central).
    # 
    # Using exp(-lambda * (cost - 1)) allows us to tune the aggressiveness.
    # lambda = 1.5 was found to improve score in similar contexts.
    # We scale the cost so that cost=1 has penalty 1.0, and cost=2 has penalty exp(-1.5) ~ 0.22.
    cost_penalty = np.exp(-1.5 * (cost - 1.0))
    
    # Combine POI and Cost Penalty
    # This acts like a weighted Thompson Sampling score.
    scores = poi * cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If we're close to the deadline, return a simpler score to save time
    if _elapsed() > _DEADLINE_SECONDS:
        # Fall back to just POI/cost without any extra computation
        pass  # Already computed
    
    return scores
# EVOLVE-BLOCK-END
