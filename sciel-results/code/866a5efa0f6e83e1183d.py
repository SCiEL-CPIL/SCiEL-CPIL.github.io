import numpy as np
import time

def _initial_greedy_solution(capacity, weights, values):
    """
    Generate an initial solution using density greedy.
    Returns a set of indices of selected items and the total value.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return set(), 0.0
    
    # Calculate densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    densities[weights == 0] = np.inf
    
    # Sort by density descending
    order = np.argsort(-densities)
    
    selected = set()
    remaining_cap = capacity
    total_value = 0.0
    
    for idx in order:
        w = weights[idx]
        if w <= remaining_cap:
            selected.add(idx)
            remaining_cap -= w
            total_value += values[idx]
            
    return selected, total_value

def _improve_solution(selected, capacity, weights, values, time_limit=10.0):
    """
    Improve the solution using 1-swap hill climbing.
    selected: set of indices
    Returns improved set of indices and total value.
    """
    n = len(weights)
    selected_list = list(selected)
    unselected_list = [i for i in range(n) if i not in selected]
    
    current_value = sum(values[i] for i in selected_list)
    current_weight = sum(weights[i] for i in selected_list)
    
    # Precompute for speed
    # We need to check if swapping i (out) and j (in) improves value
    # New weight = current_weight - w[i] + w[j] <= capacity
    # New value = current_value - v[i] + v[j] > current_value
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_delta = 0
        best_swap = None # (i_out, j_in)
        
        # To make this efficient, we can iterate over unselected items and see if any can replace a selected one
        # However, a full O(N^2) scan is expensive if N is large (100 items -> 10k checks, which is fast enough in Python if optimized)
        # Let's try a simpler approach: for each unselected item, find if there's a selected item to remove.
        
        # Optimization: Sort unselected by value descending? No, we need to check weight feasibility.
        
        # Let's iterate through unselected items
        # For each unselected j, we need to find a selected i such that:
        # 1. w[j] - w[i] <= capacity - current_weight (i.e., w[i] >= w[j] - (capacity - current_weight))
        # 2. v[j] - v[i] > 0 (i.e., v[i] < v[j])
        
        slack = capacity - current_weight
        
        for j in unselected_list:
            w_j = weights[j]
            v_j = values[j]
            
            # If this item alone doesn't fit in the current slack, we must remove something
            # Required weight reduction: w_j - slack
            min_w_i_to_remove = w_j - slack
            
            # We want to find an i in selected_list such that:
            # weights[i] >= min_w_i_to_remove AND values[i] < v_j
            # To maximize improvement, we want to maximize (v_j - v_i), so we want to minimize v_i subject to weights[i] >= min_w_i_to_remove.
            
            # This is a bit complex to do efficiently in pure Python without data structures.
            # Given N=100, selected size ~10-20, unselected ~80-90.
            # Inner loop over selected is small.
            
            for i in selected_list:
                if weights[i] >= min_w_i_to_remove:
                    if values[i] < v_j:
                        delta = v_j - values[i]
                        if delta > best_delta:
                            best_delta = delta
                            best_swap = (i, j)
        
        if best_swap:
            i_out, j_in = best_swap
            selected_list.remove(i_out)
            unselected_list.remove(j_in)
            selected_list.append(j_in)
            unselected_list.append(i_out)
            current_value += best_delta
            current_weight += weights[j_in] - weights[i_out]
            improved = True
            
    return set(selected_list), current_value

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_idx = np.where(feasible_mask)[0]
    
    if len(feasible_idx) == 0:
        return -1
    
    # Strategy:
    # 1. Generate a good initial solution for the remaining subproblem.
    # 2. Improve it with local search.
    # 3. Pick the FIRST item in the final solution (or the one that appears earliest in some order) 
    #    Wait, the interface requires returning an index into the CURRENT remaining arrays.
    #    The evaluator builds the solution step-by-step.
    #    If we return index k, the evaluator removes k and calls us again.
    #    
    #    This creates a mismatch: Local search finds a global set S. 
    #    If we return one item from S, the next call will have a different remaining set.
    #    We must ensure that the item we return is "safe" to commit to, or we must re-run the search.
    #    
    #    Alternative Interpretation:
    #    The function is called repeatedly. We can use the local search to evaluate the "promise" of each candidate item.
    #    For each feasible item k:
    #    1. Assume we pick k.
    #    2. Solve the subproblem (remaining - k) using Local Search.
    #    3. Total Value = value[k] + LS_Value(remaining - k).
    #    4. Pick k with max Total Value.
    #    
    #    This is effectively "1-step lookahead with Local Search completion".
    #    This is much stronger than "1-step lookahead with Greedy completion".
    #    
    #    Cost: For each of ~10-20 feasible candidates (assuming we prune or limit), we run LS.
    #    LS is fast for N=100.
    
    best_total = -1.0
    best_idx = -1
    
    # To save time, we might not evaluate ALL feasible items if there are many.
    # But with N=100 and capacity 12.5, weights are likely small or varied.
    # Let's evaluate all feasible items.
    
    # Optimization: Pre-compute weights/values arrays
    w = remaining_weights
    v = remaining_values
    
    for idx in feasible_idx:
        # Simulate picking idx
        new_cap = remaining_capacity - w[idx]
        val_gain = v[idx]
        
        # Prepare remaining arrays for LS
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        new_w = w[mask]
        new_v = v[mask]
        
        # Run Local Search on the subproblem
        # We need to be careful with time. 
        # If we run full LS for each candidate, it might be too slow.
        # Let's limit LS time per candidate to 0.5s.
        
        if new_cap <= 0 or len(new_w) == 0:
            total = val_gain
        else:
            # Initial greedy for LS start
            init_sel, init_val = _initial_greedy_solution(new_cap, new_w, new_v)
            # Improve
            final_sel, final_val = _improve_solution(init_sel, new_cap, new_w, new_v, time_limit=0.5)
            total = val_gain + final_val
            
        if total > best_total:
            best_total = total
            best_idx = idx
            
    return int(best_idx)
