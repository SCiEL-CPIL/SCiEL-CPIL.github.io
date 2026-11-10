"""Improved 0/1 knapsack heuristic with optimized greedy lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0
_START = None


def _greedy_fill_value_fast(weights, values, capacity, exclude_idx=None):
    """Compute total value from greedy density fill (0/1), optionally excluding one index."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        w = weights[mask]
        v = values[mask]
    else:
        w = weights
        v = values
    if len(w) == 0:
        return 0.0
    # Only consider items that fit
    fit_mask = w <= capacity
    if not np.any(fit_mask):
        return 0.0
    w = w[fit_mask]
    v = v[fit_mask]
    density = v / w
    order = np.argsort(density)[::-1]
    w_sorted = w[order]
    v_sorted = v[order]
    cum_w = np.cumsum(w_sorted)
    # Find how many items fit
    idx = np.searchsorted(cum_w, capacity, side='right') - 1
    if idx < 0:
        return 0.0
    return float(np.sum(v_sorted[:idx+1]))


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item for 0/1 knapsack using optimized greedy lookahead.
    """
    global _START
    if _START is None:
        _START = time.time()
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        # Fallback: pick highest value/weight ratio item that fits
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        fit = w <= remaining_capacity
        if not np.any(fit):
            return -1
        densities = np.zeros(len(w))
        densities[fit] = v[fit] / w[fit]
        return int(np.argmax(densities))

    n = len(remaining_weights)
    if n == 0:
        return -1
    if remaining_capacity <= 0:
        return -1

    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)

    if n == 1:
        if w[0] <= remaining_capacity:
            return 0
        return -1

    # Filter to feasible items
    fit_mask = w <= remaining_capacity
    if not np.any(fit_mask):
        return -1
    
    feasible_indices = np.where(fit_mask)[0]
    w_feas = w[fit_mask]
    v_feas = v[fit_mask]
    n_feas = len(w_feas)
    
    if n_feas == 1:
        return int(feasible_indices[0])

    densities_feas = v_feas / w_feas
    order = np.argsort(densities_feas)[::-1]
    
    # Consider top candidates by density, but limit to a reasonable number
    # With capacity 12.5, at most ~10-20 items can fit, so top 25-30 is sufficient
    k = min(n_feas, 30)
    top_k_local = order[:k]
    top_k_global = feasible_indices[top_k_local]
    
    best_score = -1.0
    best_idx = int(top_k_global[0])
    
    # Precompute sorted order for greedy fill (excluding each candidate)
    # For efficiency, we'll compute greedy value for each candidate
    
    for local_i in top_k_local:
        global_i = int(feasible_indices[local_i])
        w_i = w[global_i]
        v_i = v[global_i]
        
        rem_cap = remaining_capacity - w_i
        if rem_cap < 0:
            continue
        
        # Compute greedy fill value after taking this item
        # Exclude this item from the remaining set
        if n_feas == 1:
            greedy_val = 0.0
        else:
            # Create mask to exclude local_i
            mask = np.ones(n_feas, dtype=bool)
            mask[local_i] = False
            w_rem = w_feas[mask]
            v_rem = v_feas[mask]
            
            if len(w_rem) == 0:
                greedy_val = 0.0
            else:
                # Only consider items that fit in remaining capacity
                fit_rem = w_rem <= rem_cap
                if not np.any(fit_rem):
                    greedy_val = 0.0
                else:
                    w_fit = w_rem[fit_rem]
                    v_fit = v_rem[fit_rem]
                    dens_fit = v_fit / w_fit
                    ord_fit = np.argsort(dens_fit)[::-1]
                    w_sorted = w_fit[ord_fit]
                    v_sorted = v_fit[ord_fit]
                    cum_w = np.cumsum(w_sorted)
                    idx = np.searchsorted(cum_w, rem_cap, side='right') - 1
                    if idx < 0:
                        greedy_val = 0.0
                    else:
                        greedy_val = float(np.sum(v_sorted[:idx+1]))
        
        total_val = v_i + greedy_val
        
        # Add a small bonus for higher density items (they're more "valuable" per unit)
        # This helps break ties and prefer items that are clearly good
        density_bonus = 0.05 * densities_feas[local_i] * min(w_i, rem_cap + w_i)
        
        # Penalty for leaving a lot of unused capacity that can't be filled
        mask = np.ones(n_feas, dtype=bool)
        mask[local_i] = False
        waste_penalty = 0.01 * max(0, rem_cap - np.sum(w_feas[mask][w_feas[mask] <= rem_cap]))
        
        score = total_val + density_bonus - waste_penalty
        
        if score > best_score:
            best_score = score
            best_idx = global_i
    
    # Check deadline
    if time.time() - _START > _DEADLINE:
        return best_idx
    
    return best_idx
# EVOLVE-BLOCK-END
