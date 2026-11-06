import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    start_time = time.time()
    deadline = start_time + 140.0  # 140s margin out of 150s total

    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    C = float(remaining_capacity)

    if C <= 0:
        return -1

    # Filter feasible items
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    w_f = w[feasible_mask]
    v_f = v[feasible_mask]
    n_f = len(w_f)

    if n_f == 0:
        return -1

    # Check for zero-weight items with positive value
    zero_w_mask = (w_f == 0) & (v_f > 0)
    if np.any(zero_w_mask):
        zero_indices = np.where(zero_w_mask)[0]
        return int(feasible_indices[zero_indices[0]])

    # Compute densities for initial candidate ranking
    d_f = np.zeros(n_f)
    non_zero_w = w_f > 0
    d_f[non_zero_w] = v_f[non_zero_w] / w_f[non_zero_w]
    d_f[~non_zero_w] = 0.0

    # Sort by density descending to get top candidates
    sort_idx = np.argsort(-d_f)
    
    # Use all feasible items as candidates (evaluate each with fractional bound)
    n_candidates = n_f
    candidate_indices = np.arange(n_f)
    
    # Pre-sort by density for fractional knapsack computation
    sort_idx = np.argsort(-d_f)
    w_s = w_f[sort_idx]
    v_s = v_f[sort_idx]
    
    # Helper: compute fractional knapsack value for given capacity, excluding one item
    def fractional_bound(cap, exclude_idx):
        """
        Compute the fractional knapsack upper bound for the remaining items
        after picking the item at exclude_idx, with remaining capacity cap.
        """
        if cap <= 1e-9:
            return 0.0
        
        # Sort remaining items by density (exclude the picked item)
        # We need to skip exclude_idx in the sorted order
        total_val = 0.0
        remaining_cap = cap
        
        for i in sort_idx:
            if i == exclude_idx:
                continue
            if w_s[i] <= remaining_cap:
                total_val += v_s[i]
                remaining_cap -= w_s[i]
            else:
                # Fractional part
                if w_s[i] > 0:
                    total_val += v_s[i] * (remaining_cap / w_s[i])
                remaining_cap = 0.0
                break
            if remaining_cap <= 1e-9:
                break
        
        return total_val
    
    best_score = -1.0
    best_i_local = -1
    
    # Evaluate each candidate using Lagrangian relaxation:
    # Score = value of picked item + fractional bound of remainder
    for cand_local in candidate_indices:
        if time.time() >= deadline:
            break
        
        # Compute the score: value of this item + fractional knapsack bound for rest
        rem_cap = C - w_f[cand_local]
        if rem_cap < -1e-9:
            continue  # infeasible
        
        rem_cap = max(rem_cap, 0.0)
        frac_val = fractional_bound(rem_cap, cand_local)
        score = v_f[cand_local] + frac_val
        
        if score > best_score:
            best_score = score
            best_i_local = cand_local
    
    # If no valid candidate found, fall back to highest density feasible item
    if best_i_local == -1:
        best_i_local = candidate_indices[0]
    
    return int(feasible_indices[best_i_local])
