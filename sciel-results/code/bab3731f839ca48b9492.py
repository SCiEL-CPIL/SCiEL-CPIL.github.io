"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Convert to numpy arrays for faster operations if needed, 
    # but lists are fine for N=100.
    # We will use lists for clarity and low overhead.
    
    # Create list of indices
    indices = list(range(n))
    
    # Sort by value density descending
    # Handle zero weight (shouldn't happen, but just in case)
    def density(i):
        w = remaining_weights[i]
        if w <= 0:
            return float('inf')
        return remaining_values[i] / w
        
    indices.sort(key=density, reverse=True)
    
    best_val = -1
    best_idx = -1
    
    # 1-step lookahead: try each item as the next pick
    for i in indices:
        w_i = remaining_weights[i]
        if w_i > remaining_capacity:
            continue
            
        # Check deadline
        if time.time() > deadline:
            # Fallback to simple greedy if running out of time
            # Pick the first feasible item in density order
            for j in indices:
                if remaining_weights[j] <= remaining_capacity:
                    return j
            return -1

        current_val = remaining_values[i]
        rem_cap = remaining_capacity - w_i
        
        # Greedily fill the rest
        for j in indices:
            if i == j:
                continue
            w_j = remaining_weights[j]
            if w_j <= rem_cap:
                current_val += remaining_values[j]
                rem_cap -= w_j
            if rem_cap <= 0:
                break
                
        if current_val > best_val:
            best_val = current_val
            best_idx = i
            
    return best_idx
# EVOLVE-BLOCK-END
