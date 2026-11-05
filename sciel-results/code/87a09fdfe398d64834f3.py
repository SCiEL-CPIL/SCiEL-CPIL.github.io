"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for DP-based exact selection
_dp_state = {
    'initialized': False,
    'dp_table': None,
    'weights': None,
    'values': None,
    'n_items': 0,
    'capacity': 0,
    'selected': None,  # set of selected indices
    'start_time': None,
}


def _compute_dp_table(weights, values, capacity):
    """Compute full DP table for 0/1 knapsack."""
    n = len(weights)
    # Use integer scaling: capacity is 12.5, weights are floats
    # Scale by 10 to get integers (capacity becomes 125)
    scale = 10
    cap_int = int(round(capacity * scale))
    weights_int = np.array([int(round(w * scale)) for w in weights])
    
    # DP table: dp[i][c] = max value using items 0..i-1 with capacity c
    dp = np.zeros((n + 1, cap_int + 1), dtype=np.float64)
    for i in range(1, n + 1):
        w = weights_int[i - 1]
        v = values[i - 1]
        for c in range(cap_int + 1):
            dp[i][c] = dp[i - 1][c]
            if c >= w:
                candidate = dp[i - 1][c - w] + v
                if candidate > dp[i][c]:
                    dp[i][c] = candidate
    return dp, cap_int, scale


def _find_optimal_item(dp_table, weights_int, values, cap_int, scale, remaining_capacity, remaining_weights, remaining_values, selected_set):
    """
    Find which item is part of the optimal solution given current state.
    We trace back through the DP table to find the optimal selection,
    then pick the first unselected item in that optimal set.
    """
    n = len(weights_int)
    current_cap = int(round(remaining_capacity * scale))
    current_cap = min(current_cap, cap_int)
    
    # Trace back to find which items are in the optimal solution
    optimal_items = []
    c = current_cap
    for i in range(n, 0, -1):
        if c >= 0:
            if dp_table[i][c] != dp_table[i - 1][c]:
                # Item i-1 is in the optimal solution
                optimal_items.append(i - 1)
                c -= weights_int[i - 1]
    
    # Find the first unselected item in the optimal set
    for idx in optimal_items:
        if idx not in selected_set:
            return idx
    
    # If all optimal items are already selected, try to find any feasible item
    # that maintains optimality - pick the one with highest value density
    for idx in range(n):
        if idx not in selected_set and remaining_weights[idx] <= remaining_capacity + 1e-9:
            return idx
    
    return -1


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item to add to the knapsack, or -1 to stop.
    Uses exact DP to ensure globally optimal selections.
    """
    if _dp_state['start_time'] is None:
        _dp_state['start_time'] = time.time()
    
    # Deadline guard: 150s wall-clock, use 145s safety margin
    if time.time() - _dp_state['start_time'] > 145.0:
        return -1
    
    if not _dp_state['initialized']:
        # First call: initialize DP table
        weights = np.array(remaining_weights, dtype=np.float64)
        values = np.array(remaining_values, dtype=np.float64)
        n = len(weights)
        capacity = remaining_capacity
        
        _dp_state['weights'] = weights
        _dp_state['values'] = values
        _dp_state['n_items'] = n
        _dp_state['capacity'] = capacity
        _dp_state['selected'] = set()
        
        dp_table, cap_int, scale = _compute_dp_table(weights, values, capacity)
        _dp_state['dp_table'] = dp_table
        _dp_state['cap_int'] = cap_int
        _dp_state['scale'] = scale
        _dp_state['initialized'] = True
    
    dp_table = _dp_state['dp_table']
    cap_int = _dp_state['cap_int']
    scale = _dp_state['scale']
    weights = _dp_state['weights']
    values = _dp_state['values']
    selected_set = _dp_state['selected']
    
    # Convert current remaining to integer capacity
    current_cap = int(round(remaining_capacity * scale))
    current_cap = min(current_cap, cap_int)
    
    # Check if we can still fit any items
    if current_cap <= 0:
        return -1
    
    # Find items that fit in remaining capacity
    remaining_w = np.array(remaining_weights, dtype=np.float64)
    remaining_v = np.array(remaining_values, dtype=np.float64)
    n_remaining = len(remaining_w)
    
    if n_remaining == 0:
        return -1
    
    # Trace back through DP to find optimal items for current state
    # We need to map remaining items back to original indices
    # The remaining arrays are subsets of the original, so we need to track mapping
    
    # Actually, let's use a simpler approach: 
    # For the current state, compute which item to pick using DP traceback
    # The remaining items correspond to original items not yet selected
    
    # Get original indices of remaining items
    # remaining_weights and remaining_values are the unselected items in order
    # We need to figure out which original indices they correspond to
    
    # Since we track selected_set, the remaining items are those not in selected_set
    # in their original order
    original_indices = [i for i in range(_dp_state['n_items']) if i not in selected_set]
    
    if not original_indices:
        return -1
    
    # Now trace back through DP to find optimal selection for current capacity
    optimal_items = []
    c = current_cap
    n_orig = _dp_state['n_items']
    weights_int = np.array([int(round(w * scale)) for w in weights])
    
    for i in range(n_orig, 0, -1):
        if c >= 0:
            if dp_table[i][c] != dp_table[i - 1][c]:
                optimal_items.append(i - 1)
                c -= weights_int[i - 1]
    
    # Find the first optimal item that is in the remaining set
    remaining_set = set(original_indices)
    for idx in optimal_items:
        if idx in remaining_set:
            # Select this item
            selected_set.add(idx)
            return int(np.where(np.array(original_indices) == idx)[0][0])
    
    # If no optimal item is in remaining set (shouldn't happen normally),
    # fall back to picking the best feasible item by value density
    feasible = [i for i in range(n_remaining) if remaining_w[i] <= remaining_capacity + 1e-9]
    if not feasible:
        return -1
    
    # Pick highest value density among feasible
    densities = remaining_v[feasible] / (remaining_w[feasible] + 1e-10)
    best_local = int(np.argmax(densities))
    best_global = feasible[best_local]
    
    selected_set.add(original_indices[best_global])
    return best_global

# EVOLVE-BLOCK-END
