"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.monotonic()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit

def _fractional_bound(capacity, weights, values, exclude_idx):
    """
    Calculate the fractional knapsack upper bound for the remaining items,
    excluding the item at 'exclude_idx'.
    This is a tight upper bound for the 0/1 knapsack problem.
    """
    if capacity <= 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
        
    # Create indices of valid items (fit in capacity and not excluded)
    # We need to sort by density. 
    # To be efficient, we can just sort the valid items.
    
    # Mask for items that are not the excluded one
    # Note: We don't strictly need to filter by 'fit' here for the bound calculation 
    # logic if we handle the loop carefully, but it's cleaner to only consider items 
    # that could potentially be picked. However, fractional knapsack picks items 
    # in order of density until capacity is full. Items that don't fit individually 
    # might still be partially picked in the fractional model if they are high density? 
    # No, if weight > capacity, we can't pick even a fraction of it that fits 
    # unless we consider the fraction. But typically, we sort all, and pick 
    # fractions.
    
    # Let's create a list of (density, weight, value) for all i != exclude_idx
    # N is small (<=100), so sorting is cheap.
    
    indices = [i for i in range(n) if i != exclude_idx]
    if not indices:
        return 0.0
        
    # Calculate densities
    # Using list comprehension for clarity and speed for small N
    items = []
    for i in indices:
        w = weights[i]
        v = values[i]
        if w <= 0: continue
        items.append((v / w, w, v))
        
    if not items:
        return 0.0
        
    # Sort by density descending
    items.sort(key=lambda x: x[0], reverse=True)
    
    total_val = 0.0
    remaining_cap = capacity
    
    for density, w, v in items:
        if remaining_cap <= 0:
            break
        if w <= remaining_cap:
            total_val += v
            remaining_cap -= w
        else:
            # Take fraction
            total_val += v * (remaining_cap / w)
            remaining_cap = 0
            break
            
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select the next item to add to the knapsack using a lookahead strategy
    with fractional knapsack upper bound estimation.
    """
    # Time guard
    if time.monotonic() - _START_TIME > _TIME_LIMIT:
        return -1
        
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    # Find all feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    best_score = -1.0
    best_idx = -1
    
    # Convert to lists for faster iteration in the inner loop if needed, 
    # but numpy arrays are fine for indexing.
    # We will iterate through feasible candidates.
    
    for idx in feasible_indices:
        w_i = remaining_weights[idx]
        v_i = remaining_values[idx]
        
        # Remaining capacity after picking idx
        new_cap = remaining_capacity - w_i
        
        # Estimate value of completing the knapsack with remaining items
        # using the tighter Fractional Knapsack bound
        completion_val = _fractional_bound(new_cap, remaining_weights, remaining_values, idx)
        
        total_est_val = v_i + completion_val
        
        if total_est_val > best_score:
            best_score = total_est_val
            best_idx = idx
            
    return int(best_idx)
# EVOLVE-BLOCK-END
