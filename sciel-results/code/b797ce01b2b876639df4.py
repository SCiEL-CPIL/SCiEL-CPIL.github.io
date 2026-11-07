"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Probability of Improvement (PoI) per unit cost.
    
    PoI focuses on the likelihood of improvement rather than expected magnitude,
    which is more robust to GP uncertainty and avoids heuristic instability of
    cost-normalized EI. We use P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    
    Dividing by cost gives a cost-aware score: points that are likely to improve
    AND cheap to evaluate are preferred.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        # Return a simple score based on cost only (cheaper is better)
        return 1.0 / (cost + 1e-9)
    
    # Compute z-score for probability of improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Standard normal CDF
    # Use scipy's norm.cdf for accuracy
    from scipy.stats import norm
    poi = norm.cdf(z)
    
    # Ensure finite values
    poi = np.where(np.isfinite(poi), poi, 0.0)
    poi = np.clip(poi, 0.0, 1.0)
    
    # Cost-aware: divide by cost (higher cost = lower score)
    # cost is 1 + mean(x_scaled), so ranges from 1 to 2
    scores = poi / (cost + eps)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
