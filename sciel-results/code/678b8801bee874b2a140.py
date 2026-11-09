import numpy as np
import time

_start_time = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    UCB balances exploration and exploitation by adding a bonus proportional
    to the posterior standard deviation to the predicted mean. For minimization,
    we want to find the point with the lowest possible value. We define the
    score to maximize as the negative of the Lower Confidence Bound (LCB):
        Score = - (mu - beta * sigma) = -mu + beta * sigma
    
    The parameter beta controls the exploration-exploitation trade-off.
    We use an adaptive beta that decreases as f_best approaches the known
    minimum (0), shifting the strategy from exploration to exploitation.
    """
    if _elapsed() > _DEADLINE - _MARGIN:
        # Fallback to simple EI if time is running out, or just return zeros?
        # Better to return a valid heuristic quickly.
        # Using a simple UCB with fixed beta=1 is fast.
        beta = 1.0
        scores = -mu + beta * sigma
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    n = len(mu)
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive beta
    # f_best is the current best value (lower is better).
    # If f_best is close to 0, we are doing well, so reduce exploration.
    # If f_best is large, increase exploration.
    # Scale: Weierstrass 1D on [-0.5, 0.5] has values roughly in [0, 1].
    # Let's map f_best to a beta range, e.g., [1.0, 3.0].
    # beta = 1.0 + 2.0 * min(f_best, 1.0)
    # If f_best = 0, beta = 1.0 (exploit)
    # If f_best >= 1, beta = 3.0 (explore)
    
    if not np.isfinite(f_best):
        beta = 2.0
    else:
        # Clip f_best to [0, 1] for stability
        fb = np.clip(f_best, 0.0, 1.0)
        beta = 1.0 + 2.0 * fb
        
    scores = -mu + beta * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
