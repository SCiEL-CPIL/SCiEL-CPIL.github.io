"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Pick the next item using 1-step look-ahead with fractional knapsack bound."""
    # Deadline guard
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    if elapsed > 140.0:
        # Fallback: simple density
        n = len(remaining_weights)
        if n == 0 or remaining_capacity <= 0:
            return -1
        w = np.asarray(remaining_weights, dtype=float)
        v = np.asarray(remaining_values, dtype=float)
        safe_w = np.where(w > 0, w, 1e-9)
        densities = v / safe_w
        return int(np.argmax(densities))
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    
    # Filter to feasible items
    feasible_mask = w <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        return -1
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Precompute density for sorting
    safe_w = np.where(w > 0, w, 1e-9)
    densities = v / safe_w
    
    # For each feasible item, compute look-ahead score
    best_score = -1.0
    best_idx = int(feasible_indices[0])
    
    for i in feasible_indices:
        rem_cap = remaining_capacity - w[i]
        if rem_cap < 0:
            continue
        
        # Build the set of other items (excluding i)
        other_mask = feasible_mask.copy()
        other_mask[i] = False
        other_idxs = np.where(other_mask)[0]
        
        if len(other_idxs) == 0:
            score = v[i]
        else:
            # Sort other items by density descending
            other_densities = densities[other_idxs]
            order = other_idxs[np.argsort(-other_densities)]
            
            total_val = 0.0
            rem = rem_cap
            for j in order:
                if rem <= 0:
                    break
                if w[j] <= rem:
                    total_val += v[j]
                    rem -= w[j]
                else:
                    total_val += v[j] * (rem / w[j])
                    rem = 0
            
            score = v[i] + total_val
        
        if score > best_score:
            best_score = score
            best_idx = int(i)
    
    return best_idx
# EVOLVE-BLOCK-END
