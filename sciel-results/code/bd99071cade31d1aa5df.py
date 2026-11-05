"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using a greedy heuristic
    with a lookahead simulation to maximize total value.
    
    Args:
        remaining_capacity (float): The remaining capacity of the knapsack.
        remaining_weights (array-like): Weights of the remaining items.
        remaining_values (array-like): Values of the remaining items.
        
    Returns:
        int: Index of the selected item in the input arrays, or -1 if no item fits.
    """
    # Convert to numpy arrays for efficiency
    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)
    capacity = float(remaining_capacity)
    
    n = len(weights)
    if n == 0 or capacity <= 0:
        return -1

    # Filter out items that don't fit
    feasible_mask = weights <= capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1

    # Get weights and values for feasible items only
    feas_weights = weights[feasible_indices]
    feas_values = values[feasible_indices]
    
    # Calculate density
    # Avoid division by zero, though weights should be positive
    densities = feas_values / feas_weights
    
    # Sort feasible indices by density descending
    # argsort gives indices in sorted order. We want descending, so reverse.
    sorted_order = np.argsort(-densities)
    
    # We will evaluate the top K candidates
    # K is small to keep computation fast. 10 is a good balance.
    K = min(10, len(feasible_indices))
    
    best_val = -1
    best_original_index = -1
    
    # Pre-compute sorted weights and values for the feasible set
    # sorted_feas_weights[i] is the weight of the i-th best density item
    sorted_feas_weights = feas_weights[sorted_order]
    sorted_feas_values = feas_values[sorted_order]
    
    # Map from sorted position to original index in the input array
    # sorted_order[k] is the index in feas_weights
    # feasible_indices[feas_idx] is the index in the original weights
    # So, original_index = feasible_indices[sorted_order[k]]
    
    for k in range(K):
        # Candidate is the k-th item in the density-sorted list
        cand_pos_in_feas = sorted_order[k]
        cand_weight = feas_weights[cand_pos_in_feas]
        cand_value = feas_values[cand_pos_in_feas]
        
        # Simulate picking this candidate
        sim_capacity = capacity - cand_weight
        sim_total_value = cand_value
        
        # Fill the rest greedily with the remaining items
        # We iterate through the sorted list, skipping the candidate
        for j in range(len(sorted_order)):
            if j == k:
                continue
            w_j = sorted_feas_weights[j]
            if w_j <= sim_capacity:
                sim_total_value += sorted_feas_values[j]
                sim_capacity -= w_j
            # Optional: break if capacity is very small and min weight is larger?
            # Since sorted by density, not weight, we can't break on weight easily.
            # But if sim_capacity < min_weight_remaining, we could break.
            # For small K and N=100, this loop is fast enough.
            
        if sim_total_value > best_val:
            best_val = sim_total_value
            best_original_index = feasible_indices[cand_pos_in_feas]

    return int(best_original_index)
# EVOLVE-BLOCK-END
