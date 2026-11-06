import numpy as np
import time

_START = None
_HARD_LIMIT = 145.0  # Safety margin before 150s kill

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _fast_greedy(cap, weights, values, mode='density'):
    """
    Fast greedy heuristic to get a baseline solution.
    mode: 'density', 'value', 'weight'
    Returns: (selected_mask, total_value)
    """
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0
    
    cap = float(cap)
    selected = np.zeros(n, dtype=bool)
    total_val = 0.0
    total_w = 0.0
    
    # Create indices for sorting
    if mode == 'density':
        with np.errstate(divide='ignore', invalid='ignore'):
            score = np.where(weights > 0, values / weights, 0.0)
    elif mode == 'value':
        score = values.copy()
    elif mode == 'weight':
        # Smaller weight first (inverse)
        score = np.where(weights > 0, 1.0 / weights, 0.0)
    else:
        score = np.ones(n)
        
    order = np.argsort(-score, kind='stable')
    
    for idx in order:
        idx = int(idx)
        wi = float(weights[idx])
        if wi <= cap:
            selected[idx] = True
            cap -= wi
            total_val += float(values[idx])
            total_w += wi
            if cap <= 1e-9:
                break
            
    return selected, total_val

def _local_search_improve(selected, weights, values, cap_original):
    """
    Improves a given selection using 1-swap and 1-add moves.
    Returns improved selection and value.
    """
    n = len(weights)
    if n == 0:
        return selected, 0.0
    
    w = weights
    v = values
    cur_sel = selected.copy()
    cur_val = float(np.sum(v[cur_sel]))
    cur_w = float(np.sum(w[cur_sel]))
    rem_cap = cap_original - cur_w
    
    sel_idx = np.where(cur_sel)[0]
    un_sel_idx = np.where(~cur_sel)[0]
    
    improved = True
    # Limit iterations to avoid getting stuck in loops or taking too long
    max_iters = 100 
    
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        
        if _now() - _START > _HARD_LIMIT:
            break
            
        # 1. Try to ADD an item
        if len(un_sel_idx) > 0 and rem_cap > 1e-9:
            # Find feasible unselected items
            feasible_mask = w[un_sel_idx] <= rem_cap
            if np.any(feasible_mask):
                candidates = un_sel_idx[feasible_mask]
                # Pick the one with highest value (or density? Value is usually better for filling space)
                # Let's pick highest value density among feasible to be consistent with initial heuristic? 
                # Actually, just highest value is a strong local move.
                # Or maybe highest density? Let's try highest value first.
                best_idx = candidates[np.argmax(v[candidates])]
                
                # Check if adding improves value (it should, since v > 0)
                # But we need to ensure it doesn't block better future items? 
                # Local search assumes current state. Adding positive value is always an improvement in value.
                cur_sel[best_idx] = True
                cur_val += v[best_idx]
                rem_cap -= w[best_idx]
                
                # Update index sets
                pos = np.where(un_sel_idx == best_idx)[0][0]
                un_sel_idx = np.delete(un_sel_idx, pos)
                sel_idx = np.append(sel_idx, best_idx)
                improved = True
                continue
        
        # 2. Try to SWAP
        # For each selected item i, find unselected item j such that:
        # w[j] <= w[i] + rem_cap
        # v[j] > v[i]
        # Maximize v[j] - v[i]
        
        if len(sel_idx) > 0 and len(un_sel_idx) > 0:
            best_gain = -np.inf
            best_i = -1
            best_j = -1
            
            # Iterate over selected items
            for i in sel_idx:
                wi = w[i]
                vi = v[i]
                max_wj = wi + rem_cap
                
                # Find unselected items that fit in the space freed by i plus remaining cap
                mask_j = w[un_sel_idx] <= max_wj
                if not np.any(mask_j):
                    continue
                
                candidate_js = un_sel_idx[mask_j]
                if len(candidate_js) == 0:
                    continue
                
                # Find j with max value
                vals_j = v[candidate_js]
                best_j_local = candidate_js[np.argmax(vals_j)]
                best_val_j = v[best_j_local]
                
                gain = best_val_j - vi
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = best_j_local
            
            if best_i >= 0 and best_gain > 1e-9:
                # Perform swap
                cur_sel[best_i] = False
                cur_sel[best_j] = True
                cur_val += best_gain
                rem_cap += w[best_i] - w[best_j]
                
                # Update index sets
                pos_i = np.where(sel_idx == best_i)[0][0]
                sel_idx = np.delete(sel_idx, pos_i)
                
                pos_j = np.where(un_sel_idx == best_j)[0][0]
                un_sel_idx = np.delete(un_sel_idx, pos_j)
                
                sel_idx = np.append(sel_idx, best_j)
                
                improved = True
                continue
                
    return cur_sel, cur_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if remaining_capacity is None:
        return -1
    
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    
    if n == 0 or cap <= 0:
        return -1
        
    # Check if any item is feasible
    if not np.any(weights <= cap):
        return -1

    # 1. Generate a few diverse initial solutions using different greedy heuristics
    # We will pick the best one, then improve it.
    
    solutions = []
    
    # Density Greedy
    sel_d, val_d = _fast_greedy(cap, weights, values, 'density')
    solutions.append((sel_d, val_d))
    
    # Value Greedy
    sel_v, val_v = _fast_greedy(cap, weights, values, 'value')
    solutions.append((sel_v, val_v))
    
    # Weight Greedy (smallest first)
    sel_w, val_w = _fast_greedy(cap, weights, values, 'weight')
    solutions.append((sel_w, val_w))
    
    # 2. Improve the best initial solution
    # Pick the best one
    best_sol = max(solutions, key=lambda x: x[1])
    best_sel, best_val = best_sol
    
    # Improve it
    improved_sel, improved_val = _local_search_improve(best_sel, weights, values, cap)
    
    if improved_val > best_val:
        final_sel = improved_sel
    else:
        final_sel = best_sel
        
    selected_indices = np.where(final_sel)[0]
    
    if len(selected_indices) == 0:
        # Fallback: max density feasible
        feasible = weights <= cap
        if not np.any(feasible):
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(weights > 0, values / weights, 0.0)
        fd = np.where(feasible, density, -np.inf)
        return int(np.argmax(fd))
    
    # 3. Select the "next" item from the final optimized set.
    # Strategy: Pick the item that is most "critical" or has the highest value.
    # Since the evaluator removes the chosen item and asks again, we want to pick an item
    # that is likely to be in the optimal solution for the *remaining* items too.
    # However, simply picking the highest value item from the current optimal set is a reasonable heuristic.
    # Another idea: Pick the item with the highest density?
    # Let's stick with highest value as it directly contributes to the score.
    
    vals_in_sel = values[selected_indices]
    best_local_idx = np.argmax(vals_in_sel)
    return int(selected_indices[best_local_idx])
