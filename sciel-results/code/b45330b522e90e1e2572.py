# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s hard limit


def _greedy_fill_vectorized(capacity, weights, values, excluded_idx):
    """Simulate greedy fill by density, excluding one item. Returns total value.
    Uses a faster loop with pre-sorted indices."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    # Create mask of feasible items (excluding excluded_idx)
    mask = np.ones(n, dtype=bool)
    mask[excluded_idx] = False
    mask &= (weights <= capacity)
    if not np.any(mask):
        return 0.0
    
    idxs = np.where(mask)[0]
    w = weights[idxs]
    v = values[idxs]
    density = v / w
    
    # Sort by density descending
    order = np.argsort(-density)
    total_value = 0.0
    rem_cap = capacity
    # Use local variables for speed
    w_local = w
    v_local = v
    order_local = order
    for idx in order_local:
        wi = w_local[idx]
        if wi <= rem_cap:
            total_value += v_local[idx]
            rem_cap -= wi
            if rem_cap <= 0:
                break
    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining (feasible) arrays, or -1 to stop.
    
    Strategy:
      1. Quick feasibility check; if nothing fits, stop.
      2. Compute densities for all feasible items.
      3. 1-step look-ahead with expanded candidate pool: for each candidate,
         simulate picking it first, then greedily filling the rest by density.
         Candidates include top-density AND top-value items.
      4. Time guard: if the look-ahead is running too long, fall back to
         pure density.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    w_arr = np.asarray(remaining_weights, dtype=np.float64)
    v_arr = np.asarray(remaining_values, dtype=np.float64)
    
    # Feasibility mask
    feasible = w_arr <= remaining_capacity
    if not np.any(feasible):
        return -1
    
    feasible_idxs = np.where(feasible)[0]
    if len(feasible_idxs) == 0:
        return -1
    
    # If only one feasible item, pick it
    if len(feasible_idxs) == 1:
        return int(feasible_idxs[0])
    
    # Check time budget
    if time.time() > _DEADLINE - 5.0:
        # Not enough time for look-ahead, use pure density
        densities = v_arr[feasible_idxs] / w_arr[feasible_idxs]
        best_local = int(np.argmax(densities))
        return int(feasible_idxs[best_local])
    
    # Compute densities for feasible items
    densities_all = v_arr[feasible_idxs] / w_arr[feasible_idxs]
    
    # Expanded candidate pool: top by density + top by value (to catch high-value items)
    max_candidates = 40
    if len(feasible_idxs) <= max_candidates:
        candidates = feasible_idxs
    else:
        # Top half by density, top half by value
        n_cand = max_candidates
        n_by_density = n_cand // 2
        n_by_value = n_cand - n_by_density
        
        top_density = feasible_idxs[np.argsort(-densities_all)[:n_by_density]]
        values_feas = v_arr[feasible_idxs]
        top_value = feasible_idxs[np.argsort(-values_feas)[:n_by_value]]
        
        # Combine and deduplicate
        candidates = np.unique(np.concatenate([top_density, top_value]))
    
    best_score = -1.0
    best_idx = int(candidates[0])
    
    # Pre-sort all items by density for reuse in greedy fill
    all_densities = np.zeros(n, dtype=np.float64)
    mask_all = w_arr > 0
    all_densities[mask_all] = v_arr[mask_all] / w_arr[mask_all]
    all_densities[~mask_all] = 0.0
    density_order_all = np.argsort(-all_densities)
    
    for idx in candidates:
        # Check time
        if time.time() > _DEADLINE - 5.0:
            break
        
        w_i = w_arr[idx]
        v_i = v_arr[idx]
        rem_cap = remaining_capacity - w_i
        
        if rem_cap < 0:
            continue
        
        # Simulate greedy fill with item idx excluded
        # Use the pre-sorted density order, skipping idx and items too heavy
        total = v_i
        rc = rem_cap
        for j in density_order_all:
            if j == idx:
                continue
            wj = w_arr[j]
            if wj <= rc:
                total += v_arr[j]
                rc -= wj
                if rc <= 0:
                    break
        
        if total > best_score:
            best_score = total
            best_idx = int(idx)
    
    return best_idx
# EVOLVE-BLOCK-END
