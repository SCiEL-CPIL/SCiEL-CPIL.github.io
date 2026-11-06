import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    # For each feasible item, simulate greedy-by-density completion
    # and pick the one with best total value
    best_total = -1
    best_idx = -1
    
    # Precompute densities for speed
    densities = remaining_values / remaining_weights
    
    for idx in feasible_indices:
        # Simulate: pick item idx first, then greedy by density
        sim_cap = remaining_capacity - remaining_weights[idx]
        sim_value = remaining_values[idx]
        
        # Build a list of remaining items (excluding idx) sorted by density
        # For efficiency, we can just iterate through all items except idx
        # and greedily pick best density that fits
        
        # Create a working copy of densities and weights/values
        # To avoid allocation, we'll use a simple approach:
        # Iterate through all items, pick best density that fits, repeat
        
        used = [False] * n
        used[idx] = True
        
        # Greedy completion
        while sim_cap > 0:
            best_d = -1
            best_j = -1
            for j in range(n):
                if not used[j] and remaining_weights[j] <= sim_cap:
                    if densities[j] > best_d:
                        best_d = densities[j]
                        best_j = j
            if best_j == -1:
                break
            sim_value += remaining_values[best_j]
            sim_cap -= remaining_weights[best_j]
            used[best_j] = True
        
        if sim_value > best_total:
            best_total = sim_value
            best_idx = idx
    
    return int(best_idx)
