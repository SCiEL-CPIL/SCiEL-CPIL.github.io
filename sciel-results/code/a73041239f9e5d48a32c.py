import numpy as np
def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Lagrangian relaxation heuristic for 0/1 knapsack.
    
    We select the item maximizing v_i - lambda * w_i, where lambda is tuned
    so that the total weight of items with positive reduced cost (v_i > lambda * w_i)
    is approximately equal to the remaining capacity. This effectively solves
    a fractional knapsack relaxation to guide the 0/1 selection.
    
    If remaining_capacity is very small or no items are feasible, return -1.
    """
    # Handle edge cases: no items left
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Filter to only feasible items (weight <= remaining_capacity)
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    
    # If only one feasible item, pick it
    if len(feasible_weights) == 1:
        return int(feasible_indices[0])
        
    # Check if all feasible items fit in the remaining capacity
    total_feasible_weight = np.sum(feasible_weights)
    if total_feasible_weight <= remaining_capacity:
        # All items fit, so we can take all of them.
        # Pick the one with the highest value to maximize immediate gain.
        best_local = np.argmax(feasible_values)
        return int(feasible_indices[best_local])
        
    # Calculate densities
    densities = feasible_values / feasible_weights
    
    # Sort items by density in descending order to find the break-point item
    sort_order = np.argsort(-densities)
    sorted_weights = feasible_weights[sort_order]
    sorted_densities = densities[sort_order]
    
    # Find the item that causes the cumulative weight to exceed remaining_capacity
    # This item's density serves as the Lagrangian multiplier lambda
    cum_weight = 0.0
    lambda_val = 0.0
    for i in range(len(sorted_weights)):
        if cum_weight + sorted_weights[i] <= remaining_capacity:
            cum_weight += sorted_weights[i]
        else:
            lambda_val = sorted_densities[i]
            break
    else:
        # If the loop completes without breaking, it means all items fit,
        # which was handled above. But just in case, set lambda to the min density.
        lambda_val = sorted_densities[-1] if len(sorted_densities) > 0 else 0.0

    # Calculate reduced costs: v_i - lambda * w_i
    reduced_costs = feasible_values - lambda_val * feasible_weights
    
    # Find the item with the maximum reduced cost
    max_reduced_cost = np.max(reduced_costs)
    
    # If the best reduced cost is non-positive, it's better to stop
    if max_reduced_cost <= 0:
        return -1
        
    # Among items with the maximum reduced cost, pick the one with the highest value
    # This helps break ties effectively
    max_mask = np.isclose(reduced_costs, max_reduced_cost)
    candidates = np.where(max_mask)[0]
    
    if len(candidates) == 1:
        best_local = candidates[0]
    else:
        candidate_values = feasible_values[candidates]
        best_local = candidates[np.argmax(candidate_values)]
        
    return int(feasible_indices[best_local])
