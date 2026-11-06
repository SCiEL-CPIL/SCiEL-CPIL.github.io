"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_density_value(capacity, weights, values, exclude_idx):
    """
    Simulate the standard greedy density algorithm on the given arrays,
    excluding the item at exclude_idx.
    Returns the total value accumulated.
    """
    n = len(weights)
    if n == 0:
        return 0.0
    
    # Create a list of available indices
    available = [i for i in range(n) if i != exclude_idx]
    
    total_value = 0.0
    current_cap = capacity
    
    while available:
        # Find index in 'available' with max density that fits
        best_idx_in_avail = -1
        best_density = -1.0
        
        for i in available:
            if weights[i] <= current_cap:
                density = values[i] / weights[i]
                if density > best_density:
                    best_density = density
                    best_idx_in_avail = i
        
        if best_idx_in_avail == -1:
            break
        
        total_value += values[best_idx_in_avail]
        current_cap -= weights[best_idx_in_avail]
        available.remove(best_idx_in_avail)
        
    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic that picks the next item to add, or -1 to stop.
    Uses a 1-step lookahead: for each candidate item, simulate the rest
    of the greedy density algorithm and pick the candidate that leads
    to the highest total value.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to only feasible items (those that fit in remaining capacity)
    feasible_indices = []
    for i in range(n):
        if remaining_weights[i] <= remaining_capacity:
            feasible_indices.append(i)
    
    if not feasible_indices:
        return -1
    
    # If only one feasible item, take it
    if len(feasible_indices) == 1:
        return feasible_indices[0]
    
    # 1-step lookahead: for each feasible candidate, simulate the greedy
    # density algorithm on the remaining items and pick the best total value
    best_candidate = -1
    best_total_value = -1.0
    
    # Convert to numpy arrays for potential vectorization if needed,
    # but for small n, list operations are fine
    weights = list(remaining_weights)
    values = list(remaining_values)
    
    for cand in feasible_indices:
        # Simulate greedy density excluding cand
        total_val = _greedy_density_value(remaining_capacity, weights, values, cand)
        # Add the value of the candidate itself
        total_val += values[cand]
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_candidate = cand
    
    if best_candidate == -1:
        return -1
    
    return int(best_candidate)
# EVOLVE-BLOCK-END
