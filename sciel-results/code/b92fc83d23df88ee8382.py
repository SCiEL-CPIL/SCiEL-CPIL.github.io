"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 105.0  # safety margin before 120s kill


def _fractional_knapsack_value(capacity, weights, values, used_mask=None):
    """
    Compute the fractional knapsack upper bound using continuous relaxation.
    Sorts by density and fills fractionally. Returns total fractional value.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
    
    if used_mask is None:
        available = np.ones(n, dtype=bool)
    else:
        available = ~used_mask
    
    feasible = available & (weights > 0) & (weights <= capacity)
    if not np.any(feasible):
        return 0.0
    
    idxs = np.where(feasible)[0]
    w = weights[idxs]
    v = values[idxs]
    densities = v / w
    
    order = np.argsort(-densities)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for idx in order:
        if remaining_cap <= 0:
            break
        if w[idx] <= remaining_cap:
            total_value += v[idx]
            remaining_cap -= w[idx]
        else:
            # Take fractional part
            frac = remaining_cap / w[idx]
            total_value += frac * v[idx]
            remaining_cap = 0
            break
    
    return total_value


def _greedy_complete_fast(capacity, weights, values, used_mask=None):
    """
    Complete greedily by value density using a pre-sorted order.
    This is O(n log n) for sorting + O(n) for the fill, much faster than O(n^2) np.delete.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
    
    if used_mask is None:
        available = np.ones(n, dtype=bool)
    else:
        available = ~used_mask
    
    feasible = available & (weights <= capacity)
    if not np.any(feasible):
        return 0.0
    
    idxs = np.where(feasible)[0]
    w = weights[idxs]
    v = values[idxs]
    
    densities = v / w
    order = np.argsort(-densities)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for idx in order:
        if remaining_cap <= 0:
            break
        if w[idx] <= remaining_cap:
            total_value += v[idx]
            remaining_cap -= w[idx]
    
    return total_value


def _polish_swap(chosen_mask, weights, values, capacity, max_iters=50):
    """
    Try to improve a chosen set by swapping one chosen item for one unchosen item.
    Returns improved chosen mask.
    """
    n = len(weights)
    chosen_mask = chosen_mask.copy()
    
    if not np.any(chosen_mask):
        return chosen_mask
    
    for _ in range(max_iters):
        improved = False
        chosen_idxs = np.where(chosen_mask)[0]
        unchosen_idxs = np.where(~chosen_mask)[0]
        
        if len(unchosen_idxs) == 0:
            break
        
        # Precompute current weight sum
        current_weight = np.sum(weights[chosen_mask])
        
        for i in chosen_idxs:
            if improved:
                break
            w_out = weights[i]
            v_out = values[i]
            
            new_cap = capacity - (current_weight - w_out)
            
            candidates = unchosen_idxs[weights[unchosen_idxs] <= new_cap]
            if len(candidates) == 0:
                continue
            
            best_candidate = candidates[np.argmax(values[candidates])]
            
            if values[best_candidate] > v_out:
                chosen_mask[i] = False
                chosen_mask[best_candidate] = True
                improved = True
        
        if not improved:
            break
    
    return chosen_mask


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select the next item using fractional knapsack surrogate with lookahead and polish.
    Returns index into remaining arrays, or -1 to stop.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return -1
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    feasible = remaining_weights <= remaining_capacity
    if not np.any(feasible):
        return -1
    
    feasible_idxs = np.where(feasible)[0]
    
    if n <= 25:
        # Exhaustive evaluation for small instances
        best_score = -1
        best_idx = -1
        
        for idx in feasible_idxs:
            used = np.zeros(n, dtype=bool)
            used[idx] = True
            greedy_val = _greedy_complete_fast(
                remaining_capacity - remaining_weights[idx],
                remaining_weights,
                remaining_values,
                used_mask=used
            )
            total_val = remaining_values[idx] + greedy_val
            
            if total_val > best_score:
                best_score = total_val
                best_idx = idx
        
        return int(best_idx)
    
    else:
        # Use fractional knapsack to identify promising candidates
        # Compute density for all feasible items
        densities = remaining_values[feasible_idxs] / remaining_weights[feasible_idxs]
        
        # Sort by density descending
        sorted_local = np.argsort(-densities)
        sorted_global = feasible_idxs[sorted_local]
        
        # Take top-k candidates (k=15 for broader search)
        k = min(15, len(feasible_idxs))
        top_k_global = sorted_global[:k]
        
        # Also include items with high absolute value (not just density)
        high_value_local = np.argsort(-remaining_values[feasible_idxs])[:k//2]
        high_value_global = feasible_idxs[high_value_local]
        
        # Combine candidates
        candidate_set = set(top_k_global.tolist())
        candidate_set.update(high_value_global.tolist())
        candidates = np.array(list(candidate_set))
        
        best_score = -1
        best_idx = -1
        best_chosen = None
        
        for idx in candidates:
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            used = np.zeros(n, dtype=bool)
            used[idx] = True
            greedy_val = _greedy_complete_fast(
                remaining_capacity - remaining_weights[idx],
                remaining_weights,
                remaining_values,
                used_mask=used
            )
            total_val = remaining_values[idx] + greedy_val
            
            if total_val > best_score:
                best_score = total_val
                best_idx = idx
                best_chosen = used.copy()
        
        # Polish the best solution found
        if best_chosen is not None and time.time() - _START_TIME < _TIME_LIMIT - 5:
            polished = _polish_swap(best_chosen, remaining_weights, remaining_values, remaining_capacity)
            
            if np.sum(polished) > 0:
                polished_value = np.sum(remaining_values[polished])
                if polished_value > best_score:
                    polished_idxs = np.where(polished)[0]
                    # Prefer items from our candidates if they're in the polished set
                    for idx in candidates:
                        if polished[idx]:
                            return int(idx)
                    return int(polished_idxs[0])
        
        return int(best_idx)
# EVOLVE-BLOCK-END
