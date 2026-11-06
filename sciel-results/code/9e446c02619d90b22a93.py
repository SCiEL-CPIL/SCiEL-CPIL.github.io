import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0  # 120s margin out of 150s total
    
    # Convert to numpy arrays for vectorization
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    C = float(remaining_capacity)
    
    # Filter feasible items (weight <= capacity)
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    w_f = w[feasible_mask]
    v_f = v[feasible_mask]
    n_f = len(w_f)
    
    if n_f == 0:
        return -1
        
    # Compute densities
    d_f = np.zeros(n_f)
    non_zero_w = w_f > 0
    d_f[non_zero_w] = v_f[non_zero_w] / w_f[non_zero_w]
    d_f[~non_zero_w] = np.where(v_f[~non_zero_w] > 0, np.inf, 0.0)
    
    # Sort indices by density descending
    sort_idx = np.argsort(-d_f)
    w_s = w_f[sort_idx]
    v_s = v_f[sort_idx]
    d_s = d_f[sort_idx]
    
    # Prefix sums of weights and values in sorted order
    pw = np.cumsum(w_s)
    pv = np.cumsum(v_s)
    
    def get_lp_value(cap):
        """
        Calculate the fractional knapsack value for capacity 'cap'
        using the pre-sorted items w_s, v_s, d_s.
        """
        if cap <= 1e-9:
            return 0.0
        
        # Find the largest index k such that pw[k] <= cap
        # searchsorted returns the index where cap would be inserted to maintain order
        # side='right' means it returns index i such that a[i-1] <= cap < a[i]
        k = np.searchsorted(pw, cap, side='right') - 1
        
        if k < 0:
            # Cap is less than the weight of the first item
            # We take a fraction of the first item
            if len(w_s) == 0:
                return 0.0
            # Value is proportional to weight
            return cap * d_s[0]
        
        val = pv[k]
        remaining_cap = cap - pw[k]
        
        # If there is a next item, we take a fraction of it
        if k < len(w_s) - 1:
            next_w = w_s[k + 1]
            next_d = d_s[k + 1]
            # We can take up to next_w of the next item
            # But we only have remaining_cap capacity
            # Fractional value = remaining_cap * density
            # (Since we are in fractional mode, we assume we can take any fraction)
            val += remaining_cap * next_d
            
        return val
    
    best_score = -1.0
    best_i_local = -1
    
    # 2-step lookahead
    # We evaluate all pairs (i, j)
    for i in range(n_f):
        if w_f[i] > C:
            continue
            
        # Optimization: If current best is already very high, we might skip, 
        # but N=100 is small enough that we just run the loop.
        
        for j in range(n_f):
            if i == j:
                continue
            if w_f[i] + w_f[j] > C:
                continue
                
            # Calculate score: Value of i + Value of j + Approximate Value of Remainder
            # The remainder approximation uses the LP bound of the full set, which is a slight overestimate
            # but consistent and fast.
            rem_cap = C - w_f[i] - w_f[j]
            score = v_f[i] + v_f[j] + get_lp_value(rem_cap)
            
            if score > best_score:
                best_score = score
                best_i_local = i
                
        # Check time
        if time.time() > deadline:
            break

    # Fallback if no pair found (e.g. only 1 item fits)
    if best_i_local == -1:
        for i in range(n_f):
            if w_f[i] > C:
                continue
            rem_cap = C - w_f[i]
            score = v_f[i] + get_lp_value(rem_cap)
            if score > best_score:
                best_score = score
                best_i_local = i

    if best_i_local == -1:
        return -1

    return int(feasible_indices[best_i_local])
