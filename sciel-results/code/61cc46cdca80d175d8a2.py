import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Handle zero weight items
    zero_weight_indices = [i for i in range(n) if remaining_weights[i] == 0 and remaining_values[i] > 0]
    if zero_weight_indices:
        return zero_weight_indices[0]

    # Prepare items
    items = []
    for i in range(n):
        w = remaining_weights[i]
        v = remaining_values[i]
        if w > 0 and w <= remaining_capacity:
            items.append((v, w, i))
    
    if not items:
        return -1

    # Sort by density descending
    items.sort(key=lambda x: x[0] / x[1], reverse=True)
    
    N = len(items)
    values = np.array([x[0] for x in items])
    weights = np.array([x[1] for x in items])
    orig_indices = [x[2] for x in items]
    
    # Precompute prefix sums for B&B bound
    prefix_val = np.zeros(N + 1)
    prefix_weight = np.zeros(N + 1)
    np.cumsum(values, out=prefix_val[1:])
    np.cumsum(weights, out=prefix_weight[1:])
    
    best_val = -1
    best_first_idx = -1
    
    start_time = time.time()
    time_limit = 0.10 # 100ms per call
    
    def get_bound(i, curr_val, curr_weight):
        rem_cap = remaining_capacity - curr_weight
        if rem_cap <= 0:
            return curr_val
        
        target = prefix_weight[i] + rem_cap
        pos = np.searchsorted(prefix_weight, target, side='right') - 1
        
        if pos < i:
            return curr_val
        
        val_full = prefix_val[pos + 1] - prefix_val[i]
        weight_full = prefix_weight[pos + 1] - prefix_weight[i]
        
        total_val = curr_val + val_full
        
        if pos + 1 < N:
            rem_cap_after_full = rem_cap - weight_full
            if rem_cap_after_full > 0:
                next_idx = pos + 1
                total_val += values[next_idx] * (rem_cap_after_full / weights[next_idx])
                
        return total_val

    # B&B
    stack = [(0, 0.0, 0.0, -1)] # (i, curr_val, curr_weight, first_idx_in_path)
    # We only track the first index to save memory/time on path copying
    
    while stack:
        if time.time() - start_time > time_limit:
            break
            
        i, curr_val, curr_weight, first_idx = stack.pop()
        
        ub = get_bound(i, curr_val, curr_weight)
        if ub <= best_val:
            continue
            
        if i == N:
            if curr_val > best_val:
                best_val = curr_val
                best_first_idx = first_idx
            continue
            
        v_i = values[i]
        w_i = weights[i]
        
        # Exclude branch (push first so Include is on top)
        # If we exclude, the first_idx remains whatever it was (or -1 if not set)
        stack.append((i + 1, curr_val, curr_weight, first_idx))
        
        # Include branch
        if curr_weight + w_i <= remaining_capacity:
            new_first_idx = first_idx if first_idx != -1 else i
            stack.append((i + 1, curr_val + v_i, curr_weight + w_i, new_first_idx))

    if best_first_idx != -1:
        return orig_indices[best_first_idx]
    
    # Fallback: Lookahead Greedy (O(N^2))
    # If B&B didn't find anything or timed out, use a stronger greedy
    best_score = -1
    best_item_idx = -1
    
    for k in range(N):
        v_k = values[k]
        w_k = weights[k]
        
        rem_cap = remaining_capacity - w_k
        if rem_cap < 0:
            continue
            
        curr_val = v_k
        curr_w = w_k
        
        for j in range(N):
            if j == k:
                continue
            if curr_w + weights[j] <= remaining_capacity:
                curr_w += weights[j]
                curr_val += values[j]
                
        if curr_val > best_score:
            best_score = curr_val
            best_item_idx = k

    if best_item_idx != -1:
        return orig_indices[best_item_idx]
        
    return orig_indices[0]
