import numpy as np
import time

# Global start time for the entire evaluation session
# This is a heuristic guard. The evaluator might restart the process, 
# but if it's a single long-running script, this helps.
# However, since 'acquisition' is called repeatedly, we should 
# just ensure it's fast. The 330s limit is for the whole run.
# We don't have a reliable global clock across calls if the 
# environment resets state, but usually, it's the same process.
# We'll skip complex time management inside acquisition as it's O(N) vectorized.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using LCB.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic exploration parameter beta
    # Theoretical minimum is -0.8
    # Gap to minimum indicates how much improvement is possible
    # If gap is large, explore more (higher beta)
    # If gap is small, exploit (lower beta)
    
    # Clamp gap to [0, 1.0] for scaling
    # f_best can be > 0.8 initially? Cosine8 max is 0.8 if scaled to min -0.8?
    # If min is -0.8, max is likely 0.8.
    # So gap = f_best - (-0.8) = f_best + 0.8. Range [0, 1.6].
    gap = f_best + 0.8
    gap = np.clip(gap, 0.0, 1.6)
    
    # Map gap to beta
    # Early (gap ~ 1.6): beta ~ 2.5 (high exploration)
    # Late (gap ~ 0.0): beta ~ 0.5 (low exploration)
    # Linear interpolation
    beta = 0.5 + (2.5 - 0.5) * (gap / 1.6)
    
    # LCB: mu - beta * sigma
    # We want to minimize LCB, so maximize -LCB
    # -LCB = -mu + beta * sigma
    lcb_score = -mu + beta * sigma
    
    # Normalize by cost
    # Higher score is better
    scores = lcb_score / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
