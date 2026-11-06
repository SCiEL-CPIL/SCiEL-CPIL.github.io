"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _check_time():
    if _START_TIME is None:
        return True
    # Safety margin: use 140s out of 150s limit
    return (time.time() - _START_TIME) < 140.0


def _simulate_greedy_value(capacity, weights, values, exclude_idx):
    """Simulate greedy fill by value density, excluding one item.
    Returns total value achievable."""
    if capacity <= 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
    
    # Compute densities for all items except excluded
    if n == 1:
        return 0.0
    
    mask = np.ones(n, dtype=bool)
    mask[exclude_idx] = False
    
    w = weights[mask]
    v = values[mask]
    cap = capacity
    
    if len(w) == 0:
        return 0.0
    
    densities = v / w
    order = np.argsort(-densities)
    
    total_val = 0.0
    for idx in order:
        if cap <= 0:
            break
        wi = w[idx]
        if wi <= cap:
            cap -= wi
            total_val += v[idx]
        else:
            # Fractional part for estimation (but in 0/1 we just stop)
            # Actually in 0/1 knapsack, if it doesn't fit, we skip and try next
            pass
    
    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    _init_time()
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # If only one item, check if it fits
    if n == 1:
        if remaining_weights[0] <= remaining_capacity:
            return 0
        return -1
    
    # Filter to feasible items only
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_indices]
    feasible_values = remaining_values[feasible_indices]
    n_feasible = len(feasible_indices)
    
    # If only one feasible item
    if n_feasible == 1:
        return int(feasible_indices[0])
    
    # 1-step lookahead: for each feasible item, simulate greedy continuation
    # and pick the one that gives the best total value
    best_idx = -1
    best_score = -1.0
    
    # Compute densities for sorting in simulation
    densities = feasible_values / feasible_weights
    sort_order = np.argsort(-densities)
    
    # To speed up, we can limit the number of candidates we fully evaluate
    # based on time, but first let's try all
    # For efficiency, pre-sort and use a faster simulation
    
    # Fast simulation: sort by density, greedily pick
    # For each candidate i, the total value = v[i] + greedy_fill(cap - w[i], all except i)
    
    # Optimization: if we sort by density, the greedy fill after picking item i
    # will pick items in density order from the remaining set.
    
    # Let's do a simple approach: for each candidate, simulate
    # But this is O(n^2) in worst case. With n=100, that's 10000 ops which is fine.
    
    # Pre-compute sorted order of all feasible items by density
    sorted_idx = sort_order  # indices into feasible arrays
    
    # For each candidate, we need to simulate greedy on remaining items
    # Key insight: after picking candidate c, the remaining greedy fill
    # will pick items in density order from feasible set minus {c}
    
    # Let's build a prefix sum approach or just simulate directly
    # Direct simulation is O(n) per candidate, so O(n^2) total = 10000 for n=100
    
    for c in range(n_feasible):
        if not _check_time():
            break
        
        cap_after = remaining_capacity - feasible_weights[c]
        if cap_after < 0:
            continue
        
        # Simulate greedy fill on remaining items (all feasible except c)
        total_val = feasible_values[c]
        cap = cap_after
        
        for si in sorted_idx:
            if si == c:
                continue
            if cap <= 0:
                break
            wi = feasible_weights[si]
            if wi <= cap:
                cap -= wi
                total_val += feasible_values[si]
        
        if total_val > best_score:
            best_score = total_val
            best_idx = int(feasible_indices[c])
    
    if best_idx == -1:
        # Fallback to density
        return int(feasible_indices[np.argmax(densities)])
    
    return best_idx
# EVOLVE-BLOCK-END
