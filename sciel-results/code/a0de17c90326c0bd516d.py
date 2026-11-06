import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Ensure we have at least one feasible item (input contract says feasible, but check just in case)
    # The prompt says "remaining (feasible) arrays", so all items fit.
    
    # Calculate densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(remaining_weights > 0, remaining_values / remaining_weights, 0)
    
    # Order items by density descending
    order = np.argsort(densities)[::-1]
    
    best_score = -np.inf
    best_idx = -1
    
    for i in range(n):
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        
        cap_left = remaining_capacity - w_i
        if cap_left < 0:
            continue
            
        # Estimate value of remaining capacity using other items
        # We iterate through 'order' and pick items j != i that fit
        est_val = 0.0
        current_cap = cap_left
        
        for j in order:
            if j == i:
                continue
            w_j = remaining_weights[j]
            if w_j <= current_cap:
                est_val += remaining_values[j]
                current_cap -= w_j
            # Note: We stop if current_cap is 0? 
            # Actually, if w_j > current_cap, we can't take it fully. 
            # For a 0/1 estimate, we just skip it if it doesn't fit? 
            # Or do we take a fraction? 
            # Taking a fraction gives an upper bound. 
            # Taking only items that fully fit is a lower bound (greedy 0/1 on the rest).
            # The "greedy 0/1 on the rest" is often a better proxy for the actual 0/1 solution than fractional.
            # Let's stick to 0/1 for the estimate to be consistent with the problem type.
            
            if current_cap <= 0:
                break
                
        total_est = v_i + est_val
        
        if total_est > best_score:
            best_score = total_est
            best_idx = i
            
    return int(best_idx)
