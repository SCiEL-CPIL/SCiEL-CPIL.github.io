import numpy as np
import time

_DEADLINE = 330.0
_START_TIME = time.time()

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Strategy:
    Uses a dynamic "marginal utility" heuristic that estimates the opportunity cost
    of each item by considering the scarcity of each resource.
    
    1. For each constraint, we compute the "scarcity" as the ratio of total weight
       to capacity. Higher scarcity means the resource is more tightly contested.
    2. For each item, we compute its "marginal cost" in each constraint, weighted
       by the scarcity of that constraint. This captures how much "tight" resource
       an item consumes.
    3. The heuristic is the prize-to-weight ratio adjusted by the marginal cost,
       using a geometric mean to ensure balance across constraints.
    
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
    
    # Scarcity: how much of the capacity is "demanded"
    # If total_weight > capacity, the constraint is binding
    scarcity = np.clip(total_weights_per_constraint / capacity, 1.0, None)
    
    # Normalize scarcity weights so they sum to 1
    sum_scarcity = np.sum(scarcity)
    if sum_scarcity < eps:
        scarcity = np.ones(5)
        sum_scarcity = 5.0
    scarcity_weights = scarcity / sum_scarcity  # Shape (5,)
    
    # 3. Compute the "marginal cost" of each item in each constraint
    # This is the item's weight in that constraint, weighted by scarcity
    # An item that consumes a lot of a tight resource has high marginal cost
    # marginal_cost[i, c] = weights[i, c] * scarcity[c]
    marginal_costs = weights * scarcity[:, np.newaxis]  # Shape (5, 100)
    
    # 4. Compute a "resource efficiency" score
    # For each item, we want to reward high prize and low marginal cost
    # Use a geometric mean of (prize / marginal_cost) for each constraint
    # But marginal_cost can be zero, so we clip
    safe_marginal_costs = np.clip(marginal_costs, eps, None)
    
    # Prize-to-marginal-cost ratio for each constraint
    prize_to_cost = safe_prizes / safe_marginal_costs  # Shape (5, 100)
    
    # 5. Blend the standard ratio and the marginal cost ratio
    # Use a geometric mean to combine them
    log_ratio = np.log(ratios)  # Shape (5, 100)
    log_prize_to_cost = np.log(prize_to_cost)  # Shape (5, 100)
    
    # Blend: 50% standard ratio, 50% marginal cost ratio
    blended_log = 0.5 * log_ratio + 0.5 * log_prize_to_cost
    
    # Weighted geometric mean using scarcity weights
    weighted_log_sum = np.sum((scarcity_weights[:, np.newaxis]) * blended_log, axis=0)
    
    # Convert back to linear scale
    heuristic = np.exp(weighted_log_sum)
    
    # 6. Ensure positivity and finiteness
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # 7. Normalize to have mean 1
    mean_h = np.mean(heuristic)
    if mean_h > 0:
        heuristic = heuristic / mean_h
        
    return heuristic
