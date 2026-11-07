import numpy as np
import time

_DEADLINE = 330.0
_START_TIME = time.time()

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Strategy:
    Uses a "constraint-specific scarcity premium" heuristic.
    
    1. For each item, identify the most binding constraint (where it consumes the 
       largest fraction of its total weight).
    2. Scale the item's prize-to-weight ratio by the inverse of the remaining 
       capacity fraction for that specific constraint.
    3. This prioritizes items that offer high value relative to their most limiting 
       resource, especially when that resource is scarce.
    
    Args:
        prizes: numpy array of shape (5, 100) with prizes for each constraint/item.
        weights: numpy array of shape (5, 100) with weights for each constraint/item.
        
    Returns:
        numpy array of shape (100,) with heuristic scores (larger is more attractive).
    """
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        # Fallback to simple geometric mean if running out of time
        p = np.clip(prizes, 1e-9, None)
        w = np.clip(weights, 1e-9, None)
        ratios = p / w
        log_ratios = np.log(ratios)
        mean_log_ratio = np.mean(log_ratios, axis=0)
        heuristic = np.exp(mean_log_ratio)
        heuristic = np.maximum(heuristic, 1e-9)
        mean_h = np.mean(heuristic)
        if mean_h > 0:
            heuristic = heuristic / mean_h
        return heuristic

    # Ensure inputs are float64 arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    eps = 1e-9
    capacity = 25.0
    
    # 1. Calculate per-constraint ratios: Prize / Weight
    safe_weights = np.clip(weights, eps, None)
    safe_prizes = np.clip(prizes, eps, None)
    
    ratios = safe_prizes / safe_weights  # Shape (5, 100)
    
    # 2. Calculate scarcity for each constraint
    # Total weight of all items in each constraint
    total_weights_per_constraint = np.sum(weights, axis=1)  # Shape (5,)
    
    # Remaining capacity fraction: 1 - (total_weight / capacity)
    # If total_weight >= capacity, remaining is 0 (or negative)
    remaining_fraction = 1.0 - (total_weights_per_constraint / capacity)  # Shape (5,)
    
    # Clip to avoid division by zero or negative values
    # If remaining is <= 0, the constraint is fully utilized or over
    safe_remaining = np.clip(remaining_fraction, eps, None)
    
    # Scarcity premium: inverse of remaining fraction
    # Higher premium means the constraint is more scarce
    scarcity_premium = 1.0 / safe_remaining  # Shape (5,)
    
    # 3. For each item, find the most binding constraint
    # The most binding constraint is where the item's weight is the largest 
    # fraction of its total weight across all constraints
    total_item_weight = np.sum(weights, axis=0)  # Shape (100,)
    safe_total_item_weight = np.clip(total_item_weight, eps, None)
    
    # Fraction of total weight in each constraint for each item
    weight_fractions = weights / safe_total_item_weight[np.newaxis, :]  # Shape (5, 100)
    
    # Find the constraint with the maximum weight fraction for each item
    most_binding_constraint = np.argmax(weight_fractions, axis=0)  # Shape (100,)
    
    # 4. Get the scarcity premium for the most binding constraint of each item
    # Use advanced indexing to get the premium for each item's most binding constraint
    item_premiums = scarcity_premium[most_binding_constraint]  # Shape (100,)
    
    # 5. Compute the base heuristic: geometric mean of prize-to-weight ratios
    log_ratio = np.log(ratios)  # Shape (5, 100)
    mean_log_ratio = np.mean(log_ratio, axis=0)  # Shape (100,)
    base_heuristic = np.exp(mean_log_ratio)  # Shape (100,)
    
    # 6. Apply the scarcity premium to the base heuristic
    heuristic = base_heuristic * item_premiums  # Shape (100,)
    
    # 7. Ensure positivity and finiteness
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # 8. Normalize to have mean 1
    mean_h = np.mean(heuristic)
    if mean_h > 0:
        heuristic = heuristic / mean_h
        
    return heuristic
