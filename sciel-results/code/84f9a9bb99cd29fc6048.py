import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    
    # Calculate value-to-weight ratio (density)
    # Avoid division by zero (though weights should be > 0 in standard knapsack)
    densities = feasible_values / feasible_weights
    
    # Find the index with the maximum density
    # np.argmax returns the first occurrence in case of ties
    best_local_idx = np.argmax(densities)
    
    # Map back to the original index in the remaining arrays
    return feasible_indices[best_local_idx]
