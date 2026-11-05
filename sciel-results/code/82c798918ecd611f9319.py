import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 140.0

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        if len(remaining_weights) == 0:
            return -1
        # Fallback: Simple greedy
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        densities = v[feasible] / w[feasible]
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    w = np.array(remaining_weights)
    v = np.array(remaining_values)
    C = remaining_capacity
    
    # Pre-compute densities and sort order
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    # Handle zero weight (infinite density)
    densities[w == 0] = np.inf
    order = np.argsort(-densities, kind='stable')
    
    best_score = -np.inf
    best_idx = -1
    
    # 1-step lookahead with fractional bound
    # For each item i, estimate total value if we pick i now
    # and then fill the rest greedily (fractionally)
    
    for i in range(n):
        if w[i] > C:
            continue
            
        C_rem = C - w[i]
        frac_val = 0.0
        cur_w = 0.0
        
        # Iterate through items in density order, skipping i
        for j in order:
            if j == i:
                continue
            if cur_w + w[j] <= C_rem:
                cur_w += w[j]
                frac_val += v[j]
            else:
                # Fractional part
                if w[j] > 0:
                    frac_val += (C_rem - cur_w) * (v[j] / w[j])
                break
                
        score = v[i] + frac_val
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    if best_idx == -1:
        # No feasible item found (should not happen if any w <= C)
        feasible = np.where(w <= C)[0]
        if len(feasible) == 0:
            return -1
        # Pick highest density feasible
        densities_feasible = densities[feasible]
        best_local = np.argmax(densities_feasible)
        return int(feasible[best_local])
        
    return int(best_idx)
