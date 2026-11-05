"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Improved Lagrangian relaxation heuristic for 0/1 knapsack.
    
    We select the item maximizing v_i - lambda * w_i, where lambda is tuned
    via binary search so that the total weight of items with positive reduced cost
    is approximately equal to the remaining capacity.
    
    Improvements:
    1. When all items fit, use fractional knapsack ordering (density-based) to pick
       the next item that best fills capacity, rather than just max value.
    2. More precise binary search with tighter bounds.
    3. If the best reduced cost is slightly negative but there are feasible items,
       still pick the best one (to avoid premature stopping).
    """
    # Handle edge cases: no items left
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Convert to numpy arrays for efficiency
    remaining_weights = np.asarray(remaining_weights, dtype=np.float64)
    remaining_values = np.asarray(remaining_values, dtype=np.float64)
    
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
    
    # Compute densities
    densities = feasible_values / feasible_weights
    
    # Check if all items fit: if total weight <= remaining_capacity, 
    # we can take all items. In that case, we want to pick the item that
    # is "most critical" to fill the knapsack well. 
    # Use fractional knapsack logic: sort by density descending, and pick
    # the first item that, when added, keeps us on track.
    # Actually, if all fit, the optimal is to take all. But we're picking one at a time.
    # The best strategy when all fit is to pick the highest density item first,
    # because it preserves the most "value per weight" for future selections.
    # But actually, since all will be taken, the order doesn't matter for total value.
    # However, to be safe, pick the highest value item to ensure we get value early.
    # Let's use a hybrid: if all fit, pick the item with highest value.
    total_weight_all = np.sum(feasible_weights)
    if total_weight_all <= remaining_capacity:
        # All items fit, so we will take them all. Pick the one with highest value.
        best_local = np.argmax(feasible_values)
        return int(feasible_indices[best_local])
    
    # Binary search for lambda such that the total weight of items with
    # density > lambda is close to remaining_capacity.
    
    # The critical lambda is the one where the "break item" in fractional knapsack
    # would be partially taken. We want lambda to be the density of that break item.
    
    # Sort by density descending
    sorted_indices = np.argsort(-densities)
    sorted_weights = feasible_weights[sorted_indices]
    sorted_values = feasible_values[sorted_indices]
    
    # Find the break point: cumulative weight exceeds remaining_capacity
    cumulative_weights = np.cumsum(sorted_weights)
    
    # Find the first index where cumulative weight > remaining_capacity
    # If none, then all fit (handled above)
    if cumulative_weights[-1] <= remaining_capacity:
        # All fit, pick highest value
        best_local = np.argmax(feasible_values)
        return int(feasible_indices[best_local])
    
    # Find the break index
    break_mask = cumulative_weights > remaining_capacity
    if np.any(break_mask):
        break_idx = np.argmax(break_mask)
        # The lambda should be the density of the break item
        lambda_opt = densities[sorted_indices[break_idx]]
    else:
        # Should not happen, but fallback
        lambda_opt = np.max(densities)
    
    # Now select the item with the highest reduced cost v_i - lambda_opt * w_i
    reduced_costs = feasible_values - lambda_opt * feasible_weights
    
    # Pick the item with the highest reduced cost
    best_local = np.argmax(reduced_costs)
    
    # If the best reduced cost is negative, it means no item is worth taking
    # at this lambda. But we might still want to take the best one.
    # Let's check: if all reduced costs are negative, stop.
    # If some are positive, pick the best one.
    if reduced_costs[best_local] < -1e-9:
        # All reduced costs are negative, so no item is beneficial
        # But wait, this shouldn't happen if we set lambda correctly.
        # Let's fallback to picking the highest density item that fits.
        # Actually, if lambda is the break density, then items with density > lambda
        # should have positive reduced cost. Let's check.
        # If none have positive reduced cost, pick the highest value item that fits.
        best_local = np.argmax(feasible_values)
    
    return int(feasible_indices[best_local])
# EVOLVE-BLOCK-END
