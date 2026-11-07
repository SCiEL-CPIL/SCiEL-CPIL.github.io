import numpy as np
from scipy.stats import norm
import time

def _safe_finite(arr, default=0.0):
    """Replace NaN and Inf with a default value."""
    return np.nan_to_num(arr, nan=default, posinf=default, neginf=default)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability-Weighted Improvement.
    
    Score = P(f(x) < f_best) * (f_best - mu) / cost
    This formulation prioritizes candidates with high confidence of improvement
    and large expected gap, normalized by cost.
    """
    start_time = time.time()
    
    # Ensure inputs are float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margins for division
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in norm.cdf
    # norm.cdf is stable, but clipping ensures no overflow in intermediate steps if any
    z_clipped = np.clip(z, -10, 10)
    
    # Probability that the new point is better than f_best
    # P(f(x) < f_best) = Phi(z)
    p_improve = norm.cdf(z_clipped)
    
    # The gap in objective value
    # If mu < f_best, gap is positive (potential improvement)
    # If mu > f_best, gap is negative (worse than current best)
    gap = f_best - mu
    
    # Calculate the structural score: Probability-Weighted Improvement / Cost
    # Higher score is better
    score = (p_improve * gap) / cost_safe
    
    # Ensure all scores are finite
    score = _safe_finite(score, 0.0)
    
    # Deadline guard (though this function is O(N) vectorized and very fast)
    # If this were a more complex iterative acquisition, we would check time here.
    elapsed = time.time() - start_time
    if elapsed > 250: # Safety margin
         # Return the current score, which is already computed
         pass
        
    return score
