import numpy as np
import time

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 110.0  # Leave 10s margin for 120s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Check time
    if time.time() - START_TIME > TIME_LIMIT:
        return -1
    
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Convert to numpy arrays for fast operations if not already
    # The prompt implies they might be lists or arrays. 
    # np.sum is fast.
    
    v = np.array(remaining_values, dtype=np.float64)
    w = np.array(remaining_weights, dtype=np.float64)
    
    # If any weight is 0, density is inf. Handle separately or assume w > 0.
    # Knapsack weights are usually > 0.
    if np.any(w == 0):
        # If weight is 0, we should definitely take it if value > 0.
        # If value <= 0, skip.
        # For simplicity, assume w > 0 as per standard knapsack.
        # If w==0, v/w is inf.
        pass

    total_v = np.sum(v)
    total_w = np.sum(w)
    
    # Calculate scores for each item
    # Score_i = v_i + (C - w_i) * (total_v - v_i) / (total_w - w_i)
    
    # Avoid division by zero if total_w == w_i (i.e., only one item or all others weight 0)
    # If n==1, total_w - w_i = 0.
    
    scores = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        wi = w[i]
        vi = v[i]
        
        # If item doesn't fit, score is -inf (though prompt says feasible arrays)
        if wi > remaining_capacity:
            scores[i] = -np.inf
            continue
            
        rem_w = total_w - wi
        rem_v = total_v - vi
        
        if rem_w == 0:
            # No other items, or all other items have 0 weight
            scores[i] = vi
        else:
            # Estimated value from remaining capacity
            # We assume we can fill the remaining capacity with items of average density rem_v/rem_w
            # But we can only take up to rem_w weight of items.
            # If remaining_capacity - wi > rem_w, we can take all remaining items.
            # In that case, value is vi + rem_v.
            # If remaining_capacity - wi < rem_w, we take a fraction, approx vi + (C-wi)*(rem_v/rem_w).
            
            cap_rem = remaining_capacity - wi
            if cap_rem >= rem_w:
                # We can take all remaining items
                scores[i] = vi + rem_v
            else:
                # We take a fraction based on average density
                scores[i] = vi + cap_rem * (rem_v / rem_w)
                
    # Find index of max score
    best_idx = np.argmax(scores)
    
    if scores[best_idx] == -np.inf:
        return -1
        
    return int(best_idx)
