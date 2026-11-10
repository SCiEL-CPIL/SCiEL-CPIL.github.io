"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Uses a 'bottleneck slack' approach: for each item, compute the minimum remaining
    capacity ratio across all constraints if that item were added first, then multiply
    by the item's total prize to prioritize items that fit easily into tight constraints.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight values for each constraint/item
    
    Returns:
        array of shape (100,) - attractiveness score per item (larger is more attractive)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = weights.shape
    capacity = 25.0
    
    # Total prize per item (sum across constraints)
    total_prizes = prizes.sum(axis=0)
    
    # For each item, compute the bottleneck slack:
    # min over constraints c of (capacity - w_ic) / max(w_ic, eps)
    # This measures how much "room" is left in the tightest constraint for this item.
    # If w_ic is close to capacity, the ratio is small (bad). If w_ic is small, ratio is large (good).
    eps = 1e-9
    
    # Remaining capacity for each item in each constraint
    remaining = capacity - weights  # shape (5, 100)
    
    # Avoid division by zero: use max(w, eps)
    w_safe = np.maximum(weights, eps)
    
    # Ratio of remaining capacity to weight for each constraint/item
    ratios = remaining / w_safe  # shape (5, 100)
    
    # Bottleneck: minimum ratio across constraints for each item
    bottleneck_slack = np.min(ratios, axis=0)  # shape (100,)
    
    # Ensure non-negative (if weight > capacity, remaining is negative, ratio is negative)
    # Clip to a small positive value to maintain positivity
    bottleneck_slack = np.maximum(bottleneck_slack, eps)
    
    # Multiply by total prize to get final heuristic
    raw_heuristic = total_prizes * bottleneck_slack
    
    # Normalize the heuristic to [0, 1] to prevent extreme dominance in probability selection.
    # This encourages exploration by flattening the probability distribution slightly.
    max_heuristic = np.max(raw_heuristic)
    if max_heuristic > 0 and np.isfinite(max_heuristic):
        heuristic = raw_heuristic / max_heuristic
    else:
        heuristic = np.ones_like(raw_heuristic) * 1e-9
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
