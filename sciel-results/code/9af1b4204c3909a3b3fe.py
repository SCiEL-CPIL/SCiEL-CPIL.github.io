"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Find indices of items that fit
    feasible_indices = [i for i in range(n) if remaining_weights[i] <= remaining_capacity]
    
    if not feasible_indices:
        return -1

    best_index = -1
    best_total_value = -1

    # Helper to compute LP relaxation value for a set of items
    def compute_lp_value(cap, weights, values):
        if cap <= 0 or len(weights) == 0:
            return 0.0
        # Create list of (density, value, weight)
        items = []
        for w, v in zip(weights, values):
            if w > 0:
                items.append((v / w, v, w))
            else:
                # Zero weight item, take it all
                items.append((float('inf'), v, 0))
        
        if not items:
            return 0.0
            
        # Sort by density descending
        items.sort(key=lambda x: x[0], reverse=True)
        
        total_val = 0.0
        rem_cap = cap
        
        for density, v, w in items:
            if w == 0:
                total_val += v
            elif rem_cap >= w:
                total_val += v
                rem_cap -= w
            else:
                # Fractional part
                total_val += v * (rem_cap / w)
                rem_cap = 0
                break
                
        return total_val

    for i in feasible_indices:
        # Tentatively pick item i
        new_cap = remaining_capacity - remaining_weights[i]
        
        # Prepare remaining items (excluding i)
        rem_weights = []
        rem_values = []
        for j in range(n):
            if j != i:
                rem_weights.append(remaining_weights[j])
                rem_values.append(remaining_values[j])
        
        # Calculate LP value for the rest
        lp_val = compute_lp_value(new_cap, rem_weights, rem_values)
        total_val = remaining_values[i] + lp_val
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_index = i

    return best_index
# EVOLVE-BLOCK-END
