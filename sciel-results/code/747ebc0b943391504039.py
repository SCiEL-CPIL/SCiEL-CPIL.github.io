"""Rewritten program: 0/1 Knapsack heuristic with 1-step lookahead and greedy fill."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking to enforce deadline
_start_time = time.time()
TIME_LIMIT = 120.0  # Safety margin before 150s kill

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Deadline check
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to simple density to ensure termination
        weights = np.asarray(remaining_weights)
        values = np.asarray(remaining_values)
        if len(weights) == 0:
            return -1
        feasible = weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        feasible_indices = np.where(feasible)[0]
        feasible_weights = weights[feasible_indices]
        feasible_values = values[feasible_indices]
        densities = feasible_values / feasible_weights
        return int(feasible_indices[np.argmax(densities)])

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    n = len(weights)
    
    if n == 0:
        return -1
        
    # Identify feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    
    # If only one feasible item, return it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
        
    # Sort feasible items by density (value/weight) descending
    # We need to keep track of original indices
    densities = feasible_values / feasible_weights
    # Sort indices of the feasible array based on density
    sorted_order = np.argsort(-densities)
    
    sorted_weights = feasible_weights[sorted_order]
    sorted_values = feasible_values[sorted_order]
    sorted_orig_indices = feasible_indices[sorted_order]
    
    best_score = -1.0
    best_item_index = -1
    
    # For each candidate item, simulate picking it and then greedily filling the rest
    # We iterate through the sorted list to simulate the greedy fill
    m = len(sorted_weights)
    
    for i in range(m):
        # Candidate is the item at sorted_order[i]
        w_i = sorted_weights[i]
        v_i = sorted_values[i]
        
        # Remaining capacity and value after picking candidate
        rem_cap = remaining_capacity - w_i
        total_val = v_i
        
        # Greedily fill remaining capacity with other items
        # We iterate through the sorted list, skipping the candidate
        for j in range(m):
            if i == j:
                continue
            w_j = sorted_weights[j]
            if w_j <= rem_cap:
                rem_cap -= w_j
                total_val += sorted_values[j]
            # Since items are sorted by density, if an item doesn't fit, 
            # smaller weight items might still fit, but lower density items 
            # are less valuable per unit weight. 
            # However, a simple greedy fill just takes items in density order 
            # if they fit. It does not backtrack.
            # Optimization: If rem_cap is very small, we could break early 
            # if the smallest remaining weight is > rem_cap, but we don't have 
            # weights sorted by size. We just continue checking.
            # For N=100, this inner loop is fast.
            
        if total_val > best_score:
            best_score = total_val
            best_item_index = int(sorted_orig_indices[i])
            
    return best_item_index
# EVOLVE-BLOCK-END
