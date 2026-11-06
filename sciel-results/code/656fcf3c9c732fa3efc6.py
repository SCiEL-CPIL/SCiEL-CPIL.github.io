import numpy as np
import time

def _greedy_pack(capacity, weights, values, sort_key):
    """
    Greedily packs items using the specified sort_key.
    Returns total value.
    """
    n = len(weights)
    if n == 0 or capacity <= 1e-9:
        return 0.0
    
    if sort_key == 'density':
        d = np.where(weights > 1e-9, values / weights, np.inf)
        indices = np.argsort(-d)
    elif sort_key == 'value':
        indices = np.argsort(-values)
    elif sort_key == 'weight_asc':
        indices = np.argsort(weights)
    elif sort_key == 'value_density_ratio':
        # Sort by value / weight, but break ties by value
        d = np.where(weights > 1e-9, values / weights, np.inf)
        # Combine into a single key for sorting: primary density, secondary value
        # We want high density, then high value
        # To do this with argsort, we can sort by -density, then -value
        # But np.argsort only does one key.
        # We can create a composite score.
        # Or just sort by density.
        indices = np.argsort(-d)
    else:
        return 0.0

    total_val = 0.0
    curr_cap = capacity
    
    for idx in indices:
        w = weights[idx]
        if w <= curr_cap:
            total_val += values[idx]
            curr_cap -= w
            if curr_cap <= 1e-9:
                break
    return total_val

def _best_greedy_remainder(capacity, weights, values):
    """
    Finds the best value obtainable by greedily packing the items.
    Tries multiple sorting strategies.
    """
    n = len(weights)
    if n == 0 or capacity <= 1e-9:
        return 0.0
        
    # Filter out items that don't fit at all
    valid_mask = weights <= capacity
    if not np.any(valid_mask):
        return 0.0
        
    w_rem = weights[valid_mask]
    v_rem = values[valid_mask]
    
    if len(w_rem) == 0:
        return 0.0
        
    best_val = 0.0
    
    # Strategy 1: Density
    val1 = _greedy_pack(capacity, w_rem, v_rem, 'density')
    if val1 > best_val:
        best_val = val1
        
    # Strategy 2: Value
    val2 = _greedy_pack(capacity, w_rem, v_rem, 'value')
    if val2 > best_val:
        best_val = val2
        
    # Strategy 3: Weight ascending (fill small items first to maximize count? or just fit)
    # Actually, for knapsack, sorting by weight ascending is not usually optimal for value,
    # but might help in edge cases.
    val3 = _greedy_pack(capacity, w_rem, v_rem, 'weight_asc')
    if val3 > best_val:
        best_val = val3

    return best_val

def _dp_knapsack(capacity, weights, values):
    """
    Dynamic Programming solution for 0/1 Knapsack.
    Uses a dictionary to store reachable (weight, value) pairs to handle floating point weights.
    This is effectively a "sparse DP" or "branch and bound" hybrid.
    """
    n = len(weights)
    if n == 0:
        return 0.0
    
    # dp[w] = max value for weight exactly w? No, dp[w] = max value for weight <= w?
    # Standard DP is 1D array for integer weights.
    # For floats, we can use a dictionary: key = weight, value = max_value.
    # But weights can be many distinct values.
    # Alternatively, we can sort items by weight and use a list of (weight, value) pareto frontier.
    
    # Let's implement the Pareto Frontier approach which is efficient for small N.
    # Frontier: list of (weight, value) tuples, sorted by weight, such that value is strictly increasing.
    # If a new item (w, v) is added:
    # 1. Create new points: (w_old + w, v_old + v) for all (w_old, v_old) in frontier where w_old + w <= capacity.
    # 2. Merge new points with old points.
    # 3. Remove dominated points: a point (w1, v1) is dominated by (w2, v2) if w1 >= w2 and v1 <= v2.
    
    # Start with (0, 0)
    frontier = [(0.0, 0.0)]
    
    for i in range(n):
        w_i = weights[i]
        v_i = values[i]
        
        if w_i > capacity:
            continue
            
        # Generate new candidates
        new_points = []
        for w_f, v_f in frontier:
            w_new = w_f + w_i
            if w_new <= capacity:
                new_points.append((w_new, v_f + v_i))
        
        # Merge frontier and new_points
        # Both are sorted by weight?
        # frontier is sorted by weight.
        # new_points: since frontier is sorted by weight, new_points is also sorted by weight.
        # We need to merge two sorted lists and keep Pareto optimal points.
        
        merged = []
        i_f = 0
        i_n = 0
        
        # We want to keep points that are not dominated.
        # A point is dominated if there exists another point with <= weight and >= value.
        # Since we process in increasing weight, we can keep track of max value seen so far.
        # But we need to merge two lists.
        
        # Simpler approach:
        # 1. Concatenate frontier and new_points.
        # 2. Sort by weight.
        # 3. Iterate and keep only points where value is strictly greater than all previous points' values.
        
        all_points = frontier + new_points
        all_points.sort(key=lambda x: x[0])
        
        new_frontier = []
        max_val_so_far = -1.0
        last_weight = -1.0
        
        for w, v in all_points:
            if v > max_val_so_far:
                # Check if we have a point with same weight and higher value?
                # Sorting by weight ensures we see all same-weight points together?
                # Not necessarily if we don't break ties by value.
                # Let's sort by (weight, -value) to handle ties.
                pass
        
        # Let's re-sort properly
        all_points.sort(key=lambda x: (x[0], -x[1]))
        
        new_frontier = []
        max_val_so_far = -1.0
        
        for w, v in all_points:
            if v > max_val_so_far:
                new_frontier.append((w, v))
                max_val_so_far = v
            
            if len(new_frontier) > 10000:
                # Safety break to avoid explosion, though N is small
                break
        
        frontier = new_frontier
        
        if not frontier:
            break
            
    if not frontier:
        return 0.0
        
    return max(v for w, v in frontier)

def _exact_knapsack_mid(capacity, weights, values):
    """
    Solves 0/1 Knapsack exactly for small N using Meet in the Middle.
    """
    n = len(weights)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0] if weights[0] <= capacity else 0.0
        
    mid = n // 2
    left_w = weights[:mid]
    left_v = values[:mid]
    right_w = weights[mid:]
    right_v = values[mid:]
    
    # Generate all subsets for left
    left_subsets = []
    for mask in range(1 << mid):
        w_sum = 0.0
        v_sum = 0.0
        for i in range(mid):
            if mask & (1 << i):
                w_sum += left_w[i]
                v_sum += left_v[i]
        if w_sum <= capacity:
            left_subsets.append((w_sum, v_sum))
            
    # Generate all subsets for right
    right_subsets = []
    for mask in range(1 << (n - mid)):
        w_sum = 0.0
        v_sum = 0.0
        for i in range(n - mid):
            if mask & (1 << i):
                w_sum += right_w[i]
                v_sum += right_v[i]
        if w_sum <= capacity:
            right_subsets.append((w_sum, v_sum))
            
    if not left_subsets or not right_subsets:
        return 0.0
        
    # Sort right subsets by weight
    right_subsets.sort(key=lambda x: x[0])
    
    # Preprocess right: compute prefix max of values
    # right_subsets is sorted by weight.
    # We want max value for weight <= W.
    # So we compute an array of (weight, max_value_up_to_here).
    
    rw = []
    rv_prefix_max = []
    max_val = -1.0
    
    for w, v in right_subsets:
        if v > max_val:
            max_val = v
        rw.append(w)
        rv_prefix_max.append(max_val)
        
    rw = np.array(rw)
    rv_prefix_max = np.array(rv_prefix_max)
    
    best_total = 0.0
    for w_l, v_l in left_subsets:
        rem_cap = capacity - w_l
        # Find rightmost index where rw <= rem_cap
        idx = np.searchsorted(rw, rem_cap, side='right') - 1
        if idx >= 0:
            total_val = v_l + rv_prefix_max[idx]
            if total_val > best_total:
                best_total = total_val
                
    return best_total

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    start_time = time.time()
    deadline = start_time + 120.0  # Keep margin from 150s limit
    
    best_idx = -1
    best_total_val = -1.0
    
    # Convert to numpy arrays for speed
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    
    # Quick check: if no item fits, return -1
    if np.all(w > remaining_capacity):
        return -1

    # Pre-compute densities for sorting candidates
    # We want to evaluate the most promising candidates first to have a good best_total_val early
    # and potentially prune or just stop early if time is tight?
    # The current loop evaluates all.
    
    # Let's iterate through all items, but maybe sort by a heuristic score first?
    # If we find a very good solution early, we can't really prune the evaluation of others
    # unless we have a bound.
    # For now, just iterate.
    
    # Optimization: If n is small, we can use exact solver for the remainder.
    # Threshold for exact solver: 2^k subsets. 2^20 = 1M is too slow for meet-in-middle if done naively in python.
    # Meet in middle with 2^10 = 1024 subsets per side is fast.
    # So n <= 20 is safe for exact.
    # n <= 30 might be borderline (2^15 = 32768 subsets).
    
    exact_threshold = 24  # 2^12 = 4096 subsets per side, very fast.
    
    for i in range(n):
        w_i = w[i]
        if w_i > remaining_capacity:
            continue
            
        v_i = v[i]
        
        # Prepare remainder
        # Exclude i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        # Filter by fit
        fit_mask = w <= (remaining_capacity - w_i)
        
        valid_mask = mask & fit_mask
        
        if np.any(valid_mask):
            w_rem = w[valid_mask]
            v_rem = v[valid_mask]
            
            # Decide solver
            if len(w_rem) <= exact_threshold:
                future_val = _exact_knapsack_mid(remaining_capacity - w_i, w_rem, v_rem)
            else:
                future_val = _best_greedy_remainder(remaining_capacity - w_i, w_rem, v_rem)
        else:
            future_val = 0.0
            
        total_val = v_i + future_val
        
        if total_val > best_total_val:
            best_total_val = total_val
            best_idx = i
            
        if time.time() > deadline:
            break
            
    return int(best_idx)
