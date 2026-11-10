import numpy as np
import time

_start_time = None

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety check: if we've used 145 seconds, fallback to simple density
    if time.time() - _start_time > 145:
        if len(remaining_weights) == 0:
            return -1
        w = np.asarray(remaining_weights, dtype=float)
        v = np.asarray(remaining_values, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        densities[np.where(w == 0)] = 0
        return int(np.argmax(densities))

    n = len(remaining_weights)
    if n == 0:
        return -1
    
    if n == 1:
        return 0

    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Handle zero weights to avoid division by zero in density
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    densities[np.where(w == 0)] = 0
    
    # Sort indices by density descending
    # np.argsort returns indices that would sort the array
    # We want descending order, so we use negative densities
    order = np.argsort(-densities)
    
    best_score = -1.0
    best_idx = -1
    
    # Convert to lists for faster access in the inner loop
    w_list = w.tolist()
    v_list = v.tolist()
    order_list = order.tolist()
    
    for i in range(n):
        wi = w_list[i]
        if wi > c:
            continue # Item doesn't fit (though task says input is feasible, this is a safety check)
            
        rem_cap = c - wi
        score = v_list[i]
        
        # Calculate fractional knapsack value of remaining items
        for j in order_list:
            if j == i:
                continue
            wj = w_list[j]
            if wj <= rem_cap:
                score += v_list[j]
                rem_cap -= wj
            else:
                # Take fractional part
                if wj > 0:
                    score += v_list[j] * (rem_cap / wj)
                break
            if rem_cap <= 0:
                break
                
        if score > best_score:
            best_score = score
            best_idx = i
            
    if best_idx == -1:
        # Fallback if no item fits or something went wrong
        return int(np.argmax(densities))
        
    return best_idx
