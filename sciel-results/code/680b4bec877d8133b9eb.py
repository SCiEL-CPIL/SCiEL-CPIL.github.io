"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of the
    multiple-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100) with prize values in [0, 1]
        weights: array of shape (5, 100) with weight values in [0, 1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_items = prizes.shape[1]
    
    # Total prize for each item
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # For each item, find the constraint that is most binding (max weight)
    # and compute prize relative to that binding constraint
    max_weight = weights.max(axis=0)  # shape (100,)
    
    # Also consider the minimum prize-to-weight ratio across constraints
    # An item is only as good as its worst constraint efficiency
    eps = 1e-9
    per_constraint_eff = prizes / (weights + eps)  # shape (5, 100)
    min_eff = per_constraint_eff.min(axis=0)  # shape (100,)
    
    # Primary metric: total prize / max weight
    # This captures "how much total value do I get per unit of my most expensive constraint?"
    primary = total_prize / (max_weight + eps)
    
    # Secondary: minimum per-constraint efficiency
    # This ensures the item isn't terrible in any single constraint
    
    # Combine: use a power-weighted geometric mean
    # Strongly emphasize the primary metric since it's most directly relevant
    log_primary = np.log(primary + eps)
    log_min_eff = np.log(min_eff + eps)
    
    # Weighted combination: 70% primary, 30% min efficiency
    combined = 0.7 * log_primary + 0.3 * log_min_eff
    heuristic = np.exp(combined)
    
    # Normalize to a reasonable range to avoid numerical issues
    # Scale so that the median is around 1.0
    median_val = np.median(heuristic)
    if median_val > eps:
        heuristic = heuristic / median_val
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
