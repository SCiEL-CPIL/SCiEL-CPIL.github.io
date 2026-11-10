import numpy as np
import time
import heapq

_GLOBAL_DEADLINE = None
_GLOBAL_ELAPSED = 0.0
_GLOBAL_CALL_COUNT = 0

def _check_deadline():
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        return True
    return (time.time() - _GLOBAL_ELAPSED) < (_GLOBAL_DEADLINE - 3.0)  # 3s safety margin

def _solve_knapsack_bnb(capacity, weights, values):
    """
    Branch and bound for 0/1 knapsack.
    Returns (optimal_value, list_of_indices_included).
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    # Sort by value density descending
    idx_sorted = np.argsort(-values / weights)
    sorted_w = weights[idx_sorted].astype(np.float64)
    sorted_v = values[idx_sorted].astype(np.float64)
    
    # Precompute prefix sums for fast fractional bound
    prefix_w = np.zeros(n + 1)
    prefix_v = np.zeros(n + 1)
    prefix_w[1:] = np.cumsum(sorted_w)
    prefix_v[1:] = np.cumsum(sorted_v)
    
    # Fractional bound using binary search:
    # Find the last item we can fully take, then fractionally take the next
    def fractional_bound(start_idx, rem_cap):
        if start_idx >= n or rem_cap <= 0:
            return 0.0
        
        target = prefix_w[start_idx] + rem_cap
        
        # Binary search for the rightmost j where prefix_w[j] <= target
        lo = start_idx
        hi = n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if prefix_w[mid] <= target:
                lo = mid
            else:
                hi = mid - 1
        
        val_full = prefix_v[lo] - prefix_v[start_idx]
        
        if lo < n:
            remaining_after_full = rem_cap - (prefix_w[lo] - prefix_w[start_idx])
            if remaining_after_full > 0 and sorted_w[lo] > 0:
                val_full += sorted_v[lo] * (remaining_after_full / sorted_w[lo])
        
        return val_full
    
    # Initial greedy solution with multiple strategies to get tight lower bound
    def greedy_multi_strategy():
        best_gv = 0.0
        best_gi = []
        
        # Strategy 1: density order
        gv = 0.0
        gi = []
        cap = capacity
        for i in range(n):
            if sorted_w[i] <= cap:
                gv += sorted_v[i]
                cap -= sorted_w[i]
                gi.append(i)
        if gv > best_gv:
            best_gv = gv
            best_gi = list(gi)
        
        # Strategy 2: value order (take highest value items that fit)
        val_order = np.argsort(-sorted_v)
        gv = 0.0
        gi = []
        cap = capacity
        for i in val_order:
            if sorted_w[i] <= cap:
                gv += sorted_v[i]
                cap -= sorted_w[i]
                gi.append(int(i))
        if gv > best_gv:
            best_gv = gv
            best_gi = list(gi)
        
        # Strategy 3: weight order (take lightest items first)
        wt_order = np.argsort(sorted_w)
        gv = 0.0
        gi = []
        cap = capacity
        for i in wt_order:
            if sorted_w[i] <= cap:
                gv += sorted_v[i]
                cap -= sorted_w[i]
                gi.append(int(i))
        if gv > best_gv:
            best_gv = gv
            best_gi = list(gi)
        
        # Strategy 4: density but skip items that are too heavy relative to remaining cap
        gv = 0.0
        gi = []
        cap = capacity
        for i in range(n):
            if sorted_w[i] <= cap and sorted_w[i] <= cap * 0.5:
                gv += sorted_v[i]
                cap -= sorted_w[i]
                gi.append(i)
        if gv > best_gv:
            best_gv = gv
            best_gi = list(gi)
        
        return best_gv, best_gi
    
    greedy_val, greedy_indices = greedy_multi_strategy()
    
    best_value = greedy_val
    best_items = list(greedy_indices)
    
    # B&B with max-heap
    # Use a list to track included items instead of frozenset for speed
    heap = []
    root_bound = fractional_bound(0, capacity)
    heapq.heappush(heap, (-root_bound, 0, 0.0, 0.0, []))
    
    max_nodes = 100000
    
    node_count = 0
    last_deadline_check = 0
    
    while heap and max_nodes > 0:
        node_count += 1
        # Check deadline every 1000 nodes to reduce overhead
        if node_count - last_deadline_check >= 1000:
            last_deadline_check = node_count
            if not _check_deadline():
                break
        
        max_nodes -= 1
        neg_bound, idx, val, wt, included = heapq.heappop(heap)
        bound = -neg_bound
        
        if bound <= best_value:
            continue
        
        if idx >= n:
            if val > best_value:
                best_value = val
                best_items = list(included)
            continue
        
        # Branch 1: Include item idx (try include first for better pruning)
        if sorted_w[idx] <= capacity - wt:
            new_wt = wt + sorted_w[idx]
            new_val = val + sorted_v[idx]
            new_included = included + [idx]
            next_idx = idx + 1
            if next_idx >= n:
                if new_val > best_value:
                    best_value = new_val
                    best_items = list(new_included)
            else:
                bound_next = new_val + fractional_bound(next_idx, capacity - new_wt)
                if bound_next > best_value:
                    heapq.heappush(heap, (-bound_next, next_idx, new_val, new_wt, new_included))
        
        # Branch 2: Exclude item idx
        next_idx = idx + 1
        if next_idx >= n:
            if val > best_value:
                best_value = val
                best_items = list(included)
        else:
            bound_next = val + fractional_bound(next_idx, capacity - wt)
            if bound_next > best_value:
                heapq.heappush(heap, (-bound_next, next_idx, val, wt, included))
    
    # Map back to original indices
    result_indices = [int(idx_sorted[i]) for i in best_items]
    return best_value, result_indices


def _greedy_density(capacity, weights, values):
    """Multi-strategy greedy for fallback."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    best_total = 0.0
    best_result = []
    
    # Strategy 1: density order
    idx_sorted = np.argsort(-values / weights)
    total = 0.0
    cap = capacity
    result = []
    for i in idx_sorted:
        if weights[i] <= cap:
            total += values[i]
            cap -= weights[i]
            result.append(int(i))
    if total > best_total:
        best_total = total
        best_result = list(result)
    
    # Strategy 2: value order
    val_order = np.argsort(-values)
    total = 0.0
    cap = capacity
    result = []
    for i in val_order:
        if weights[i] <= cap:
            total += values[i]
            cap -= weights[i]
            result.append(int(i))
    if total > best_total:
        best_total = total
        best_result = list(result)
    
    # Strategy 3: weight order
    wt_order = np.argsort(weights)
    total = 0.0
    cap = capacity
    result = []
    for i in wt_order:
        if weights[i] <= cap:
            total += values[i]
            cap -= weights[i]
            result.append(int(i))
    if total > best_total:
        best_total = total
        best_result = list(result)
    
    return best_total, best_result


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _GLOBAL_DEADLINE, _GLOBAL_ELAPSED, _GLOBAL_CALL_COUNT
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_idx = np.where(feasible_mask)[0]
    
    if len(feasible_idx) == 0:
        return -1
    
    # Set up global deadline on first call
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_ELAPSED = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_ELAPSED + 145.0  # 145s out of 150s budget
        _GLOBAL_CALL_COUNT = 0
    
    _GLOBAL_CALL_COUNT += 1
    
    # For later calls, use faster heuristic to save time
    # The first few calls can afford full B&B, later calls should be quick
    if _GLOBAL_CALL_COUNT <= 3:
        # Full B&B for first few calls
        try:
            opt_val, opt_indices = _solve_knapsack_bnb(remaining_capacity, remaining_weights, remaining_values)
        except Exception:
            opt_val, opt_indices = _greedy_density(remaining_capacity, remaining_weights, remaining_values)
        
        if len(opt_indices) == 0:
            _, greedy_indices = _greedy_density(remaining_capacity, remaining_weights, remaining_values)
            if len(greedy_indices) == 0:
                return -1
            return greedy_indices[0]
        
        # Pick the item from the optimal solution that maximizes value density
        best_density = -1
        best_choice = opt_indices[0]
        for idx in opt_indices:
            dens = remaining_values[idx] / remaining_weights[idx]
            if dens > best_density:
                best_density = dens
                best_choice = idx
        
        return int(best_choice)
    else:
        # Later calls: use faster multi-strategy greedy
        _, greedy_indices = _greedy_density(remaining_capacity, remaining_weights, remaining_values)
        if len(greedy_indices) == 0:
            return -1
        
        # Pick highest density item from greedy solution
        best_density = -1
        best_choice = greedy_indices[0]
        for idx in greedy_indices:
            dens = remaining_values[idx] / remaining_weights[idx]
            if dens > best_density:
                best_density = dens
                best_choice = idx
        
        return int(best_choice)
