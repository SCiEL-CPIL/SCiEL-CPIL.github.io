# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're too close to the 150s limit, fall back to simple density
    elapsed = time.time() - _start_time
    if elapsed > 140.0:
        return int(np.argmax(remaining_values / remaining_weights))
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    remaining_weights = np.asarray(remaining_weights, dtype=np.float64)
    remaining_values = np.asarray(remaining_values, dtype=np.float64)
    
    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    # Get feasible indices in the original remaining array
    feasible_indices = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_mask]
    fv = remaining_values[feasible_mask]
    fn = len(fw)
    
    if fn == 0:
        return -1
    
    # Compute Lagrangian multiplier λ
    # Strategy: binary search for λ such that the total weight of items with v_i - λ*w_i > 0
    # approximately equals remaining_capacity
    # But this is complex; use a simpler heuristic:
    # λ = total_value / total_weight (average density) as a starting point
    # Then adjust based on capacity scarcity
    
    total_w = fw.sum()
    total_v = fv.sum()
    
    if total_w <= 0:
        return int(feasible_indices[0])
    
    # Estimate λ: the "marginal" density
    # Sort by density, find the item that would be the last one we can fit
    densities = fv / fw
    sorted_idx = np.argsort(densities)[::-1]  # descending density
    sorted_w = fw[sorted_idx]
    sorted_v = fv[sorted_idx]
    
    # Find the point where cumulative weight exceeds capacity
    cum_w = np.cumsum(sorted_w)
    # The marginal item is where cum_w first exceeds remaining_capacity
    if cum_w[-1] <= remaining_capacity:
        # All items fit, pick highest value
        return int(feasible_indices[sorted_idx[np.argmax(sorted_v)]])
    
    overflow_idx = np.searchsorted(cum_w, remaining_capacity, side='right')
    if overflow_idx == 0:
        overflow_idx = 1
    
    # The marginal density is around the item at overflow_idx
    # Use a weighted estimate
    if overflow_idx < len(sorted_w):
        lambda_est = sorted_v[overflow_idx] / sorted_w[overflow_idx]
    else:
        lambda_est = total_v / total_w
    
    # Refine: use the average density of items we would likely take
    # Items with density > lambda_est are "likely taken"
    likely_mask = densities > lambda_est * 0.8
    if np.sum(likely_mask) > 0:
        lambda_refined = fv[likely_mask].sum() / fw[likely_mask].sum()
        lambda_est = 0.5 * lambda_est + 0.5 * lambda_refined
    
    # Now score items by Lagrangian: v_i - λ*w_i
    lag_scores = fv - lambda_est * fw
    
    # Get top-K candidates by Lagrangian score
    K = min(8, fn)
    top_k_local = np.argsort(lag_scores)[::-1][:K]
    
    # For each candidate, simulate the full greedy remainder
    best_candidate = -1
    best_total_value = -1
    
    for cand_local in top_k_local:
        # Tentatively pick this item
        if fw[cand_local] > remaining_capacity:
            continue
        
        rem_cap = remaining_capacity - fw[cand_local]
        rem_value = fv[cand_local]
        
        # Build the remaining set (exclude cand_local)
        mask = np.ones(fn, dtype=bool)
        mask[cand_local] = False
        rem_w = fw[mask]
        rem_v = fv[mask]
        
        # Greedily pick remaining items using Lagrangian scoring
        while rem_cap > 0 and len(rem_w) > 0:
            # Filter feasible
            feas = rem_w <= rem_cap
            if not np.any(feas):
                break
            fw_rem = rem_w[feas]
            fv_rem = rem_v[feas]
            
            # Compute Lagrangian scores for remaining
            lag_rem = fv_rem - lambda_est * fw_rem
            best_local = np.argmax(lag_rem)
            
            rem_cap -= fw_rem[best_local]
            rem_value += fv_rem[best_local]
            
            # Remove the picked item
            rem_w = np.delete(rem_w, best_local)
            rem_v = np.delete(rem_v, best_local)
        
        if rem_value > best_total_value:
            best_total_value = rem_value
            best_candidate = feasible_indices[cand_local]
    
    if best_candidate == -1:
        # Fallback: pick highest Lagrangian score
        best_local = np.argmax(lag_scores)
        best_candidate = feasible_indices[best_local]
    
    return int(best_candidate)
# EVOLVE-BLOCK-END
