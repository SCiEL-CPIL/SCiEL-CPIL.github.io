import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    Uses relative cost normalization and adaptive penalty.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)

    # 1. Compute UCB for minimization
    # We want to minimize f. The Lower Confidence Bound is LCB = mu - beta * sigma.
    # We want LCB to be small.
    # So the "score" (higher is better) is -LCB = -mu + beta * sigma.
    beta = 2.0
    ucb_scores = -mu + beta * sigma
    
    # Shift UCB scores to be non-negative to avoid negative scores
    ucb_min = np.min(ucb_scores)
    ucb_scores = ucb_scores - ucb_min
    
    # 2. Compute Relative Cost Normalization
    # This makes the penalty adaptive to the current batch of candidates
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-10
    cost_norm = (cost - cost_min) / cost_range  # [0, 1]
    cost_norm = np.clip(cost_norm, 0.0, 1.0)
    
    # 3. Adaptive Penalty Strength
    # Increase penalty as we converge (f_best -> 0)
    # Assume function range is roughly [0, 5]
    convergence = max(0.0, min(1.0, 1.0 - f_best / 5.0))
    alpha = 0.2 + 0.5 * convergence  # [0.2, 0.7]
    
    # 4. Apply Exponential Cost Penalty
    penalty = np.exp(-alpha * cost_norm)
    score = ucb_scores * penalty
    
    # 5. Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() > deadline:
        pass
        
    return score
