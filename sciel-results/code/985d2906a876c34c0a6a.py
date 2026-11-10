import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    n = len(w)
    
    if n == 0:
        return -1
        
    # Calculate densities
    # Avoid division by zero, though weights should be > 0
    densities = v / w
    
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities)
    
    best_score = -1
    best_idx = -1
    
    for i in range(n):
        # Item i is the candidate to pick
        new_cap = remaining_capacity - w[i]
        
        if new_cap < 0:
            # Should not happen if input is feasible, but safe check
            continue
            
        # Calculate fractional value of the rest
        frac_val = 0.0
        current_cap = new_cap
        
        for j in sorted_indices:
            if j == i:
                continue
            if current_cap <= 0:
                break
            if w[j] <= current_cap:
                frac_val += v[j]
                current_cap -= w[j]
            else:
                # Fractional part
                frac_val += v[j] * (current_cap / w[j])
                current_cap = 0
                break
                
        total_score = v[i] + frac_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
    return int(best_idx)
