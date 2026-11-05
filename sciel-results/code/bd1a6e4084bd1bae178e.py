# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0  # 10s safety margin before 140s hard limit


def _greedy_fill(capacity, weights, values, excluded_idx):
    """Simulate greedy fill by density, excluding one item. Returns total value."""
    if capacity <= 0:
        return 0.0
    # Create mask of feasible items (excluding excluded_idx)
    mask = np.ones(len(weights), dtype=bool)
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
    for idx in order:
        if w[idx] <= rem_cap:
            total_value += v[idx]
            rem_cap -= w[idx]
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
      3. 1-step look-ahead: for each feasible candidate, simulate picking it
         first, then greedily filling the rest by density. Pick the candidate
         that yields the highest total simulated value.
      4. Time guard: if the look-ahead is running too long, fall back to
         pure density.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Feasibility mask
    feasible = remaining_weights <= remaining_capacity
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
        densities = remaining_values[feasible_idxs] / remaining_weights[feasible_idxs]
        best_local = int(np.argmax(densities))
        return int(feasible_idxs[best_local])
    
    # Look-ahead: for each feasible candidate, simulate picking it + greedy fill
    # To keep it fast, limit the number of candidates we evaluate
    # If there are too many, sample the top ones by density
    max_candidates = 30
    densities_all = remaining_values[feasible_idxs] / remaining_weights[feasible_idxs]
    
    if len(feasible_idxs) > max_candidates:
        # Pick top max_candidates by density to evaluate
        top_order = np.argsort(-densities_all)[:max_candidates]
        candidates = feasible_idxs[top_order]
    else:
        candidates = feasible_idxs
    
    best_score = -1.0
    best_idx = int(candidates[0])
    
    w_arr = np.asarray(remaining_weights, dtype=np.float64)
    v_arr = np.asarray(remaining_values, dtype=np.float64)
    
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
        # Only consider items that fit in rem_cap
        mask = (w_arr <= rem_cap)
        mask[idx] = False
        if not np.any(mask):
            total = v_i
        else:
            idxs2 = np.where(mask)[0]
            w2 = w_arr[idxs2]
            v2 = v_arr[idxs2]
            density2 = v2 / w2
            order2 = np.argsort(-density2)
            total = v_i
            rc = rem_cap
            for j in order2:
                if w2[j] <= rc:
                    total += v2[j]
                    rc -= w2[j]
                if rc <= 0:
                    break
        
        if total > best_score:
            best_score = total
            best_idx = int(idx)
    
    return best_idx
# EVOLVE-BLOCK-END
