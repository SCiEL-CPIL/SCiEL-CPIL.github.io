# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Pareto-frontier based cost-aware acquisition function.
    
    Strategy:
    - Compute a predicted improvement score (based on LCB) and evaluation cost.
    - Perform a non-dominated sort on (improvement, cost) to identify Pareto layers.
    - Score candidates by their rank in the Pareto layers, with higher scores for
      points on the efficient frontier (high improvement, low cost).
    - Within each layer, further rank by marginal improvement per unit cost.
    - This avoids the suppression of expensive-but-promising regions inherent in
      simple cost-normalized scalarization.
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip sigma and cost to avoid numerical issues
    sigma = np.clip(sigma, 1e-6, 1e6)
    cost = np.clip(cost, 0.5, 100.0)
    
    # Hyperparameter for LCB: beta controls exploration/exploitation
    beta = 2.0
    
    # Compute predicted improvement: we want to maximize potential improvement over f_best
    # LCB = mu - beta * sigma (we want to minimize this)
    # Improvement = f_best - LCB = f_best - mu + beta * sigma
    improvement = f_best - mu + beta * sigma
    
    # Clip improvement to be non-negative for meaningful ranking
    improvement = np.maximum(improvement, 0.0)
    
    # We now have two objectives:
    # 1. Maximize improvement
    # 2. Minimize cost
    # Non-dominated sort: a point A dominates B if A.improvement >= B.improvement 
    # and A.cost <= B.cost, with at least one strict inequality.
    
    n = len(mu)
    
    # For efficiency, we'll do a simplified Pareto ranking:
    # Sort by improvement descending, then for each point, check if it's dominated
    # by any point with higher or equal improvement and lower or equal cost.
    
    # Efficient approach: sort by improvement descending, then maintain min cost seen so far
    # A point is on the Pareto front if its cost is less than the minimum cost of all 
    # points with higher or equal improvement.
    
    # Sort indices by improvement descending
    order = np.argsort(-improvement, kind='mergesort')
    
    # Track the minimum cost seen so far (among points with higher or equal improvement)
    min_cost_so_far = np.inf
    pareto_layer = np.zeros(n, dtype=np.int64)
    
    # First pass: identify Pareto front (layer 0)
    for idx in order:
        if cost[idx] < min_cost_so_far:
            pareto_layer[idx] = 0
            min_cost_so_far = cost[idx]
        else:
            pareto_layer[idx] = 1  # Temporarily assign to layer 1
    
    # For simplicity and speed, we'll use a two-layer approach:
    # Layer 0: Pareto front
    # Layer 1: All others
    
    # Score: higher is better
    # For layer 0 (Pareto front), score by improvement/cost ratio
    # For layer 1, give a lower base score, also scaled by improvement/cost
    
    # Compute improvement/cost ratio
    ratio = improvement / cost
    
    # Base score for Pareto front
    score = np.zeros(n, dtype=np.float64)
    
    # Layer 0: Pareto front - high score
    layer0_mask = (pareto_layer == 0)
    if np.any(layer0_mask):
        # Normalize ratio within layer 0 to [0, 1]
        r0 = ratio[layer0_mask]
        if len(r0) > 1:
            r0_min = np.min(r0)
            r0_max = np.max(r0)
            if r0_max > r0_min:
                r0_norm = (r0 - r0_min) / (r0_max - r0_min)
            else:
                r0_norm = np.ones_like(r0)
        else:
            r0_norm = np.ones_like(r0)
        
        # High base score for Pareto front
        score[layer0_mask] = 10.0 + r0_norm
        
    # Layer 1: Non-Pareto - lower score
    layer1_mask = (pareto_layer == 1)
    if np.any(layer1_mask):
        r1 = ratio[layer1_mask]
        if len(r1) > 1:
            r1_min = np.min(r1)
            r1_max = np.max(r1)
            if r1_max > r1_min:
                r1_norm = (r1 - r1_min) / (r1_max - r1_min)
            else:
                r1_norm = np.ones_like(r1)
        else:
            r1_norm = np.ones_like(r1)
        
        # Lower base score for non-Pareto
        score[layer1_mask] = 5.0 + r1_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
