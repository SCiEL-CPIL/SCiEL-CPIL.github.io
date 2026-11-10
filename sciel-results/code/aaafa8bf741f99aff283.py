"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fractional_knapsack_value(capacity, weights, values, exclude_idx=-1):
    """
    Calculate the upper bound value using fractional knapsack heuristic.
    Items are assumed to be sorted by density (value/weight) in descending order 
    for efficiency, or we sort here. 
    To be safe and simple, we sort here.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    # Create list of (density, weight, value) excluding the specified index
    items = []
    for i in range(len(weights)):
        if i == exclude_idx:
            continue
        if weights[i] > 0:
            items.append((values[i] / weights[i], weights[i], values[i]))
        else:
            # If weight is 0, we take it fully if capacity allows (which it always does for 0 weight)
            # But usually weights are > 0.
            items.append((float('inf'), weights[i], values[i]))
            
    if not items:
        return 0.0
        
    # Sort by density descending
    items.sort(key=lambda x: x[0], reverse=True)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for density, w, v in items:
        if remaining_cap <= 0:
            break
        if w <= remaining_cap:
            total_value += v
            remaining_cap -= w
        else:
            # Take fraction
            total_value += v * (remaining_cap / w)
            remaining_cap = 0
            break
            
    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    best_score = -1
    best_idx = -1
    
    # Pre-compute densities for potential tie-breaking or simple checks
    # But the main logic relies on the lookahead.
    
    for i in range(n):
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        
        # Check feasibility (though input is supposed to be feasible)
        if w_i > remaining_capacity:
            continue
            
        new_cap = remaining_capacity - w_i
        
        # Estimate the best value we can get from the remaining items
        # using the fractional relaxation as a proxy for the optimal 0/1 solution
        # We exclude item i from this estimation
        future_value = _fractional_knapsack_value(new_cap, remaining_weights, remaining_values, exclude_idx=i)
        
        total_estimated_value = v_i + future_value
        
        # Tie-breaking: If estimated values are very close, prefer higher density 
        # or smaller weight to preserve flexibility? 
        # For now, just max estimated value.
        if total_estimated_value > best_score:
            best_score = total_estimated_value
            best_idx = i
            
    if best_idx == -1:
        return -1
        
    return int(best_idx)
# EVOLVE-BLOCK-END
