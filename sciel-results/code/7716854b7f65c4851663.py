"""Rewritten heuristic for ant-colony multi-constraint knapsack."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Return one attractiveness score per item (larger = more attractive).
    
    Strategy:
    - prizes shape (5, 100), weights shape (5, 100), all in [0,1].
    - 5 constraints, each with capacity 25.
    - We want items that are good candidates: high prize, low weight, and
      that don't over-consume any single constraint.
    
    Approach:
    1. Compute per-item total prize (sum over constraints).
    2. Compute per-item total weight (sum over constraints).
    3. Compute per-item max weight across constraints (worst-case constraint load).
    4. Combine these into a composite score that rewards high prize-to-weight
       ratio while penalizing items that are heavy in any single constraint.
    5. Add small diversity signals from different ratio formulations.
    6. Normalize and ensure positivity.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    capacity = 25.0
    
    # Basic per-item statistics
    total_prize = prizes.sum(axis=0)       # (n_items,)
    total_weight = weights.sum(axis=0)     # (n_items,)
    max_weight = weights.max(axis=0)       # (n_items,) - worst constraint
    min_weight = weights.min(axis=0)       # (n_items,) - best constraint
    mean_weight = weights.mean(axis=0)     # (n_items,)
    
    # Safety: avoid division by zero
    eps = 1e-12
    
    # Primary ratio: total prize / total weight
    ratio1 = total_prize / (total_weight + eps)
    
    # Ratio with max weight (penalizes items that blow up one constraint)
    ratio2 = total_prize / (max_weight + eps)
    
    # Ratio with mean weight
    ratio3 = total_prize / (mean_weight + eps)
    
    # Geometric mean ratio: balances all constraints
    # Use product of (capacity - weight_c) / weight_c for each constraint, 
    # which is large when item leaves lots of room in all constraints
    # But simpler: use 1/max_weight as a proxy for "room left"
    inv_max = 1.0 / (max_weight + eps)
    
    # Composite: weight the different ratios
    # ratio2 is most important because it penalizes constraint violations
    composite = (
        0.4 * ratio2 / (ratio2.max() + eps) +
        0.3 * ratio1 / (ratio1.max() + eps) +
        0.2 * ratio3 / (ratio3.max() + eps) +
        0.1 * inv_max / (inv_max.max() + eps)
    )
    
    # Also consider: items with high prize but low weight in ALL constraints
    # are especially valuable. Add a bonus for low max_weight.
    # Normalize max_weight so lower is better
    min_mw = max_weight.min()
    max_mw = max_weight.max()
    if max_mw - min_mw > eps:
        low_weight_bonus = (max_mw - max_weight) / (max_mw - min_mw)
    else:
        low_weight_bonus = np.ones(n_items)
    
    composite += 0.3 * low_weight_bonus
    
    # Ensure all values are strictly positive
    composite = np.maximum(composite, 1e-9)
    
    # Normalize to [0, 1] range for stability
    cmin = composite.min()
    cmax = composite.max()
    if cmax - cmin > eps:
        composite = (composite - cmin) / (cmax - cmin)
    else:
        composite = np.ones(n_items)
    
    # Add small floor to ensure no zero values
    composite = composite + 1e-9
    
    return composite
# EVOLVE-BLOCK-END
