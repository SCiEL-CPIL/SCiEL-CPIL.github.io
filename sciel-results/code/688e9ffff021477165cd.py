import numpy as np
import time
import random

# Global timer
_start_time = time.time()
_TIMEOUT = 140.0 # Safety margin before 150s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Simple greedy
        if len(remaining_weights) == 0:
            return -1
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        densities = v[feasible] / w[feasible]
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Convert to numpy arrays for speed
    w = np.array(remaining_weights)
    v = np.array(remaining_values)
    C = remaining_capacity
    
    # 1. Initial Greedy Solution
    # Sort by density (v/w) descending
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    # Items with 0 weight have infinite density, put them first
    order = np.argsort(-densities, kind='stable')
    
    current_indices = []
    current_value = 0.0
    current_weight = 0.0
    available = set(range(n))
    
    for idx in order:
        if w[idx] <= C - current_weight:
            current_indices.append(idx)
            current_value += v[idx]
            current_weight += w[idx]
            available.remove(idx)
            
    if not current_indices:
        return -1

    best_solution = current_indices
    best_value = current_value
    
    # 2. Local Search
    num_iters = 0
    max_iters = 50 # Cap on local search steps to ensure speed
    
    while num_iters < max_iters:
        improved = False
        sol_set = set(current_indices)
        out_set = available
        
        # Sort outside items by density to prioritize good swaps
        out_indices = list(out_set)
        if out_indices:
            out_densities = densities[out_indices]
            sorted_out = [out_indices[i] for i in np.argsort(-out_densities)]
        else:
            sorted_out = []
            
        for i_in in current_indices:
            if improved:
                break
            for i_out in sorted_out:
                # Check if swap is valid
                new_weight = current_weight - w[i_in] + w[i_out]
                if new_weight <= C:
                    new_value = current_value - v[i_in] + v[i_out]
                    if new_value > best_value + 1e-9:
                        # Accept swap
                        current_indices.remove(i_in)
                        current_indices.append(i_out)
                        current_weight = new_weight
                        current_value = new_value
                        sol_set.remove(i_in)
                        sol_set.add(i_out)
                        out_set.remove(i_out)
                        out_set.add(i_in)
                        best_solution = current_indices
                        best_value = new_value
                        improved = True
                        break
            if improved:
                break
        num_iters += 1
        if improved:
            continue
        else:
            # No 1-for-1 improvement. Try 2-for-1 swaps?
            # If no 1-for-1, try to remove 2 items and add 1 better item.
            # This is O(K^2 * N). K is small.
            found_21 = False
            # Sort solution items by weight ascending? Or just iterate.
            # We want to free up space to add a high value item.
            # Iterate over outside items (sorted by density/value).
            for i_out in sorted_out:
                if found_21:
                    break
                # Check if adding i_out requires removing 2 items
                # Space needed: w[i_out] - (C - current_weight)
                # We need to remove items with total weight >= w[i_out] - (C - current_weight)
                # And maximize value removed? No, we want to minimize value removed.
                # Actually, we want to check if there exist i_a, i_b in solution such that:
                # current_weight - w[i_a] - w[i_b] + w[i_out] <= C
                # And current_value - v[i_a] - v[i_b] + v[i_out] > best_value
                
                # Simplified: Just try all pairs? K is ~10-20. K^2 is 100-400.
                # N is 100. Total 40k ops. Fast.
                
                # But we only need to check if it's beneficial.
                # Let's just do a limited check.
                
                # If we don't have time, skip 2-for-1.
                if time.time() - _start_time > _TIMEOUT - 5:
                     break
                
                # Optimization: Only consider i_out if it has high value.
                # We already sorted by density.
                
                # Check pairs
                sol_list = list(current_indices)
                k = len(sol_list)
                for a_idx in range(k):
                    i_a = sol_list[a_idx]
                    for b_idx in range(a_idx + 1, k):
                        i_b = sol_list[b_idx]
                        
                        new_weight = current_weight - w[i_a] - w[i_b] + w[i_out]
                        if new_weight <= C:
                            new_value = current_value - v[i_a] - v[i_b] + v[i_out]
                            if new_value > best_value + 1e-9:
                                # Accept 2-for-1 swap
                                current_indices.remove(i_a)
                                current_indices.remove(i_b)
                                current_indices.append(i_out)
                                current_weight = new_weight
                                current_value = new_value
                                sol_set.remove(i_a)
                                sol_set.remove(i_b)
                                sol_set.add(i_out)
                                out_set.remove(i_out)
                                out_set.add(i_a)
                                out_set.add(i_b)
                                best_solution = current_indices
                                best_value = new_value
                                improved = True
                                found_21 = True
                                break
                    if found_21:
                        break
                if found_21:
                    break
            
            if improved:
                continue
            else:
                 # No improvement found in 1-for-1 or 2-for-1
                 break

    # Select an item from the best solution found
    if not best_solution:
        return -1
        
    # To ensure we don't get stuck in a loop or pick a bad item, 
    # pick the item with highest density from the best solution
    sol_arr = np.array(best_solution)
    sol_densities = densities[sol_arr]
    best_pick_rel = np.argmax(sol_densities)
    return int(sol_arr[best_pick_rel])
