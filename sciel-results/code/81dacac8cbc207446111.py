"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a Pareto-frontier selection strategy based on non-dominated sorting
    of (improvement, uncertainty, cost).
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline
    if time.time() - t0 > DEADLINE - 10:
        # Fallback to simple cost-aware UCB
        beta = 1.0
        improvement = f_best - mu + beta * sigma
        scores = improvement / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute base metrics
    # 1. Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Uncertainty (standard deviation)
    uncertainty = sigma
    
    # 3. Cost (to be minimized)
    # We want to minimize cost, so in non-dominated sorting for "minimization" of cost,
    # lower cost is better.
    
    # Normalize metrics to comparable scales for sorting? 
    # Actually, non-dominated sorting doesn't strictly require normalization if we just check dominance,
    # but scales can affect how "dominant" a point feels in terms of rank if we were doing distance.
    # Here we just do strict dominance: A dominates B if A is <= B in all objectives and < in at least one.
    # Objectives to minimize:
    #   1. -EI (we want max EI, so min -EI)
    #   2. -Uncertainty (we want high uncertainty for exploration? Or is uncertainty a cost?)
    #   3. Cost (we want min cost)
    
    # Let's define the objectives to minimize:
    # obj1 = -ei (maximize improvement)
    # obj2 = -uncertainty (maximize uncertainty/exploration potential)
    # obj3 = cost (minimize cost)
    
    obj1 = -ei
    obj2 = -uncertainty
    obj3 = cost
    
    n = len(mu)
    
    # Efficient non-dominated sorting (Front 0)
    # A point is in Front 0 if no other point dominates it.
    # Dominance: i dominates j if obj1[i] <= obj1[j] and obj2[i] <= obj2[j] and obj3[i] <= obj3[j]
    # and at least one is strictly less.
    
    # For N=2000, O(N^2) is 4,000,000 operations, which is very fast in NumPy if vectorized,
    # but explicit loops might be slow. Let's try a vectorized approach for Front 0.
    
    # To check if point i is dominated by any point j:
    # (obj1[j] <= obj1[i]) & (obj2[j] <= obj2[i]) & (obj3[j] <= obj3[i]) & 
    # (obj1[j] < obj1[i] | obj2[j] < obj2[i] | obj3[j] < obj3[i])
    
    # This is O(N^2) memory if we create a full matrix (2000x2000 = 4M floats = 32MB). This is acceptable.
    
    # Construct matrices
    # We want a mask D[i, j] = True if i dominates j.
    # Or simpler: Is j dominated?
    
    # Let's compute for each point j, is there an i such that i dominates j?
    
    # To save memory and time, we can sort by one objective and then check.
    # Sort by obj1 (improvement). If i dominates j, obj1[i] <= obj1[j].
    # So we only need to check i's with obj1[i] <= obj1[j].
    
    # However, a simpler vectorized check for "is dominated" using broadcasting might be risky for memory if N was larger, but for 2000 it's fine.
    # Let's use the broadcasting method.
    
    # O1 = obj1[:, None] vs obj1[None, :]
    # A dominates B if A_obj <= B_obj for all 3, and A_obj < B_obj for at least one.
    
    # Let's create arrays of shape (N, N)
    # O1_i = obj1[:, np.newaxis]
    # O1_j = obj1[np.newaxis, :]
    
    # dom_mask[i, j] = (O1_i <= O1_j) & (O2_i <= O2_j) & (O3_i <= O3_j) & ~((O1_i == O1_j) & (O2_i == O2_j) & (O3_i == O3_j))
    
    # We want to know for each j (column), if there exists an i (row) such that dom_mask[i, j] is True.
    # dominated = np.any(dom_mask, axis=0)
    
    # This creates a 2000x2000 boolean matrix. 4 million booleans = 4 MB. Very safe.
    
    O1_i = obj1[:, np.newaxis]
    O1_j = obj1[np.newaxis, :]
    O2_i = obj2[:, np.newaxis]
    O2_j = obj2[np.newaxis, :]
    O3_i = obj3[:, np.newaxis]
    O3_j = obj3[np.newaxis, :]
    
    # Check dominance
    # i dominates j
    le_1 = O1_i <= O1_j
    le_2 = O2_i <= O2_j
    le_3 = O3_i <= O3_j
    
    # Strictly less in at least one
    lt_1 = O1_i < O1_j
    lt_2 = O2_i < O2_j
    lt_3 = O3_i < O3_j
    
    dominates = (le_1 & le_2 & le_3) & (lt_1 | lt_2 | lt_3)
    
    # A point j is dominated if any i dominates it.
    # dominates has shape (N, N). dominates[i, j] is True if i dominates j.
    # We want to know for each j, if np.any(dominates[:, j]).
    # This is equivalent to: dominated = np.any(dominates, axis=0)
    
    dominated = np.any(dominates, axis=0)
    
    # Front 0 (Pareto Frontier) consists of non-dominated points.
    is_front0 = ~dominated
    
    # Assign scores
    # Base score: Cost-aware EI (or similar)
    # We want to strongly prefer Front 0 points.
    
    # Let's use a base score that is reasonable for all points, then boost Front 0.
    # Base score: EI / cost
    base_score = ei / cost
    
    # Boost for Front 0
    # How much to boost?
    # If we just add a large constant, we might ignore the relative quality within the front.
    # Better: Score = (Rank-based bonus) + (Normalized Base Score)
    
    # Or: Score = Base Score * (1 + Boost * is_front0)
    # But Front 0 points might have low EI if they are very low cost but low improvement?
    # No, Front 0 is defined by (High EI, High Uncertainty, Low Cost).
    # So Front 0 points are inherently "good" in a multi-objective sense.
    
    # Let's calculate a normalized score within the front to preserve ordering.
    # And a normalized score for non-front points.
    
    # Simple approach:
    # Score = Base Score
    # If Front 0: Score += Large Constant (e.g., max(Base Score) * 0.5)
    
    max_base = np.max(base_score) if np.any(np.isfinite(base_score)) else 1.0
    if not np.isfinite(max_base) or max_base == 0:
        max_base = 1.0
        
    # To ensure Front 0 points are always selected over non-Front 0 points,
    # we can add a bonus that is larger than the range of base scores.
    bonus = max_base * 0.5
    
    scores = base_score.copy()
    scores[is_front0] += bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() - t0 > DEADLINE - 5:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
