import sys
import time
import numpy as np

sys.setrecursionlimit(2000)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    start_time = time.time()
    time_limit = 0.8
    
    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)
    n = len(weights)
    
    if n == 0 or remaining_capacity <= 0:
        return -1

    feasible_indices = np.where(weights <= remaining_capacity)[0]
    if len(feasible_indices) == 0:
        return -1
    
    items = []
    for i in feasible_indices:
        items.append((weights[i], values[i], int(i)))
    
    if len(items) == 1:
        return items[0][2]

    # Sort by density descending
    def get_density(item):
        w, v, idx = item
        if w == 0:
            return float('inf') if v > 0 else 0
        return v / w
        
    items.sort(key=get_density, reverse=True)
    
    # Greedy initialization
    best_val = 0
    best_set = []
    curr_cap = remaining_capacity
    curr_val = 0
    greedy_set = []
    for w, v, idx in items:
        if w <= curr_cap:
            curr_cap -= w
            curr_val += v
            greedy_set.append(idx)
        else:
            break
    if curr_val > best_val:
        best_val = curr_val
        best_set = greedy_set

    n_items = len(items)
    
    suffix_value = np.zeros(n_items + 1)
    suffix_weight = np.zeros(n_items + 1)
    for i in range(n_items - 1, -1, -1):
        suffix_value[i] = suffix_value[i+1] + items[i][1]
        suffix_weight[i] = suffix_weight[i+1] + items[i][0]

    def fractional_bound(k, cap):
        if k >= n_items:
            return 0
        if suffix_weight[k] <= cap:
            return suffix_value[k]
        total_val = 0
        for i in range(k, n_items):
            w, v, idx = items[i]
            if w <= cap:
                total_val += v
                cap -= w
            else:
                if w > 0:
                    total_val += v * (cap / w)
                cap = 0
                break
        return total_val

    timeout_occurred = [False]

    def solve(k, cap, val, chosen):
        nonlocal best_val, best_set, timeout_occurred
        
        if timeout_occurred[0]:
            return
            
        if val + fractional_bound(k, cap) <= best_val:
            return
            
        if k == n_items:
            if val > best_val:
                best_val = val
                best_set = chosen
            return

        if time.time() - start_time > time_limit:
            timeout_occurred[0] = True
            return

        w, v, idx = items[k]

        if w <= cap:
            solve(k + 1, cap - w, val + v, chosen + [idx])
            
        if not timeout_occurred[0]:
            solve(k + 1, cap, val, chosen)

    solve(0, remaining_capacity, 0, [])

    if best_set:
        return min(best_set)
    
    return -1
