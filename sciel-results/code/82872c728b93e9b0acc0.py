import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    
    # Filter to feasible items only
    feasible = weights <= remaining_capacity
    if not np.any(feasible):
        return -1
    
    f_weights = weights[feasible]
    f_values = values[feasible]
    f_indices = np.where(feasible)[0]
    
    # Compute densities
    densities = f_values / f_weights
    
    # Find the fractional knapsack break point to estimate lambda
    order = np.argsort(densities)[::-1]  # descending density
    sorted_w = f_weights[order]
    sorted_v = f_values[order]
    cum_w = np.cumsum(sorted_w)
    
    # Find where cumulative weight exceeds remaining_capacity
    if cum_w[0] > remaining_capacity:
        # Even the best density item alone exceeds capacity (shouldn't happen since we filtered feasible)
        # Actually if single item weight <= capacity, cum_w[0] <= capacity
        # So this means the first item fits
        break_idx = 0
    else:
        # Find first index where cum_w > remaining_capacity
        exceed = np.where(cum_w > remaining_capacity)[0]
        if len(exceed) == 0:
            # All items fit, just pick highest value
            break_idx = len(sorted_w) - 1
        else:
            break_idx = exceed[0]
    
    # Lambda = density of the item at break point
    if break_idx < len(densities):
        lam = densities[order[break_idx]]
    else:
        lam = densities[order[-1]]
    
    # Score each feasible item by v - lambda * w
    scores = f_values - lam * f_weights
    
    # Pick the best scored item
    best_local = int(np.argmax(scores))
    return int(f_indices[best_local])
