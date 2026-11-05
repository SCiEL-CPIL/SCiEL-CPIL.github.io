"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_fill(weights, values, capacity, order):
    """Fill knapsack greedily using items in the given order."""
    total_value = 0.0
    remaining_cap = capacity
    for idx in order:
        w = weights[idx]
        if w <= remaining_cap:
            total_value += values[idx]
            remaining_cap -= w
            if remaining_cap <= 0:
                break
    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using a multi-criteria lookahead
    heuristic with time-bounded local search refinement.
    
    Args:
        remaining_capacity (float): The remaining capacity of the knapsack.
        remaining_weights (array-like): Weights of the remaining items.
        remaining_values (array-like): Values of the remaining items.
        
    Returns:
        int: Index of the selected item in the input arrays, or -1 if no item fits.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
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
    
    n_feas = len(feasible_indices)
    
    # Calculate multiple sorting criteria
    densities = feas_values / feas_weights
    
    # Sort by different criteria
    order_density = np.argsort(-densities)
    order_value = np.argsort(-feas_values)
    order_weight = np.argsort(feas_weights)  # Lightest first
    order_value_weight = np.argsort(-feas_values / (feas_weights + 0.001))  # Value with slight weight penalty
    
    # We will evaluate the top K candidates from each criterion
    K = min(15, n_feas)
    
    best_val = -1
    best_original_index = -1
    
    # Evaluate candidates from density-sorted list
    for k in range(K):
        if time.time() > deadline:
            break
            
        cand_pos_in_feas = order_density[k]
        cand_weight = feas_weights[cand_pos_in_feas]
        cand_value = feas_values[cand_pos_in_feas]
        
        # Simulate picking this candidate and fill rest greedily by density
        sim_capacity = capacity - cand_weight
        sim_total_value = cand_value
        
        for j in order_density:
            if j == cand_pos_in_feas:
                continue
            w_j = feas_weights[j]
            if w_j <= sim_capacity:
                sim_total_value += feas_values[j]
                sim_capacity -= w_j
                if sim_capacity <= 0:
                    break
            
        if sim_total_value > best_val:
            best_val = sim_total_value
            best_original_index = feasible_indices[cand_pos_in_feas]

    # Evaluate candidates from value-sorted list (top few)
    K_val = min(5, n_feas)
    for k in range(K_val):
        if time.time() > deadline:
            break
            
        cand_pos_in_feas = order_value[k]
        cand_weight = feas_weights[cand_pos_in_feas]
        cand_value = feas_values[cand_pos_in_feas]
        
        # Simulate picking this candidate and fill rest greedily by density
        sim_capacity = capacity - cand_weight
        sim_total_value = cand_value
        
        for j in order_density:
            if j == cand_pos_in_feas:
                continue
            w_j = feas_weights[j]
            if w_j <= sim_capacity:
                sim_total_value += feas_values[j]
                sim_capacity -= w_j
                if sim_capacity <= 0:
                    break
            
        if sim_total_value > best_val:
            best_val = sim_total_value
            best_original_index = feasible_indices[cand_pos_in_feas]

    # Local search refinement: try swapping the chosen item with nearby alternatives
    if best_original_index != -1 and time.time() < deadline:
        best_feas_idx = feasible_indices.tolist().index(best_original_index)
        best_feas_pos = np.where(feasible_indices == best_original_index)[0][0]
        
        # Try a few nearby swaps in the density order
        density_rank = np.where(order_density == best_feas_pos)[0][0]
        
        for delta in [-2, -1, 1, 2, 3]:
            if time.time() > deadline:
                break
                
            new_rank = density_rank + delta
            if 0 <= new_rank < n_feas:
                alt_feas_pos = order_density[new_rank]
                alt_weight = feas_weights[alt_feas_pos]
                alt_value = feas_values[alt_feas_pos]
                
                # Simulate picking the alternative
                sim_capacity = capacity - alt_weight
                sim_total_value = alt_value
                
                for j in order_density:
                    if j == alt_feas_pos:
                        continue
                    w_j = feas_weights[j]
                    if w_j <= sim_capacity:
                        sim_total_value += feas_values[j]
                        sim_capacity -= w_j
                        if sim_capacity <= 0:
                            break
                
                if sim_total_value > best_val:
                    best_val = sim_total_value
                    best_original_index = feasible_indices[alt_feas_pos]

    return int(best_original_index)
# EVOLVE-BLOCK-END
