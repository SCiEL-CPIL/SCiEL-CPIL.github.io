"""Pareto-frontier based cost-aware acquisition for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE_SECONDS = 320.0  # 330s limit, 10s safety margin
_START_TIME = None


def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _check_deadline():
    """Return True if we should stop (deadline exceeded)."""
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE_SECONDS


def _fast_ei(mu, sigma, f_best):
    """Fast Expected Improvement score."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier ranking.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: float, lowest observed value
    - cost: array of shape (2000,), evaluation costs
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    _ensure_start()
    
    if _check_deadline():
        # Fallback: simple cost-aware EI
        ei = _fast_ei(mu, sigma, f_best)
        score = ei / (cost + 1e-9)
        return np.where(np.isfinite(score), score, 0.0)
    
    n = len(mu)
    eps = 1e-12
    
    # Compute Expected Improvement (exploitation signal)
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute Lower Confidence Bound for exploration (lower is better for minimization)
    # LCB = mu - beta * sigma; we want low LCB values
    # Use adaptive beta based on position
    beta = 2.0
    lcb = mu - beta * sigma
    
    # Two objectives to minimize:
    # Objective 1: predicted value (use a combination of mu and LCB)
    # Objective 2: evaluation cost
    #
    # For the predicted value objective, blend mu and LCB:
    # pred_value = alpha * mu + (1 - alpha) * LCB
    # alpha = 0.5 for balanced exploration/exploitation
    alpha = 0.5
    pred_value = alpha * mu + (1.0 - alpha) * lcb
    
    # Now we have two objectives to minimize:
    # obj1 = pred_value (lower is better)
    # obj2 = cost (lower is better)
    
    # Compute non-dominated sort (fast, using vectorized approach)
    # A point i dominates point j if:
    #   pred_value[i] <= pred_value[j] AND cost[i] <= cost[j] AND (at least one is strict)
    
    # For efficiency with n=2000, we can compute ranks using a simplified approach:
    # Rank by cost first, then within similar cost levels, rank by pred_value
    
    # Step 1: Sort by cost
    cost_order = np.argsort(cost)
    
    # Step 2: Compute non-dominated frontier and ranks
    # For each point, count how many points dominate it
    # This is O(n^2) in worst case but n=2000 is manageable
    
    # Actually, let's use a more efficient approach:
    # Compute a composite score based on Pareto rank
    
    # First, normalize both objectives to [0, 1] range
    pv_min, pv_max = np.min(pred_value), np.max(pred_value)
    c_min, c_max = np.min(cost), np.max(cost)
    
    pv_range = pv_max - pv_min + eps
    c_range = c_max - c_min + eps
    
    pv_norm = (pred_value - pv_min) / pv_range  # 0 = best, 1 = worst
    c_norm = (cost - c_min) / c_range           # 0 = best, 1 = worst
    
    # Compute dominance count for each point
    # Point i dominates point j if pv_norm[i] <= pv_norm[j] and c_norm[i] <= c_norm[j] and not both equal
    # For efficiency, use a grid-based approach or just compute directly for n=2000
    
    # Direct computation: for each point, count dominators
    # This is O(n^2) = 4 million operations, which is fine for n=2000
    
    # Vectorized dominance count
    # For point i, it is dominated by point j if:
    #   pv_norm[j] <= pv_norm[i] AND c_norm[j] <= c_norm[i] AND (pv_norm[j] < pv_norm[i] OR c_norm[j] < c_norm[i])
    
    # Let's compute the number of points that dominate each point
    # Using broadcasting would use too much memory (2000x2000 = 4M floats = 32MB, actually fine)
    
    pv_diff = pv_norm[:, None] - pv_norm[None, :]  # pv_diff[i,j] = pv_norm[i] - pv_norm[j]
    c_diff = c_norm[:, None] - c_norm[None, :]      # c_diff[i,j] = c_norm[i] - c_norm[j]
    
    # j dominates i if pv_diff[i,j] >= 0 and c_diff[i,j] >= 0 and not (both == 0)
    dominates = (pv_diff >= 0) & (c_diff >= 0) & ~((pv_diff == 0) & (c_diff == 0))
    
    # Count how many points dominate each point i
    dominance_count = dominates.sum(axis=1)
    
    # Points on the Pareto frontier have dominance_count = 0
    # Higher dominance_count means more dominated (worse)
    
    # Score: inverse of (dominance_count + 1) gives higher score to frontier points
    # Also incorporate the EI as a tiebreaker / additional signal
    
    # Base score from Pareto rank
    pareto_score = 1.0 / (dominance_count + 1.0)
    
    # Normalize pareto_score to [0, 1]
    ps_max = np.max(pareto_score) + eps
    ps_min = np.min(pareto_score)
    ps_range = ps_max - ps_min + eps
    pareto_score_norm = (pareto_score - ps_min) / ps_range
    
    # Combine with cost-aware EI for additional discrimination
    ei_cost = ei / (cost + eps)
    ei_cost_max = np.max(ei_cost) + eps
    ei_cost_norm = ei_cost / ei_cost_max
    
    # Final score: blend of Pareto-frontier score and cost-aware EI
    # Weight more towards Pareto score since that's the structural change
    w_pareto = 0.7
    w_ei = 0.3
    
    score = w_pareto * pareto_score_norm + w_ei * ei_cost_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
