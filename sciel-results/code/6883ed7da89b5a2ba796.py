import numpy as np
import time

def _simulate_greedy_value_fast(capacity, weights, values, sorted_indices, exclude_set):
    """
    Simulates greedy packing using a pre-sorted list of indices by density.
    Skips items in exclude_set.
    """
    total_value = 0.0
    current_cap = capacity
    
    for idx in sorted_indices:
        if idx in exclude_set:
            continue
        w = weights[idx]
        if w > current_cap:
            continue
        total_value += values[idx]
        current_cap -= w
        if current_cap <= 1e-9:
            break
            
    return total_value

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # 100s margin
    
    # Pre-compute densities and sort indices
    densities = np.zeros(n)
    for i in range(n):
        if remaining_weights[i] > 0:
            densities[i] = remaining_values[i] / remaining_weights[i]
        else:
            densities[i] = float('inf') if remaining_values[i] > 0 else 0.0
            
    sorted_indices = np.argsort(-densities).tolist()  # Descending order, as list for faster membership
    
    # Pre-filter: only consider items that fit in remaining capacity
    feasible_indices = [i for i in range(n) if remaining_weights[i] <= remaining_capacity]
    if not feasible_indices:
        return -1
    
    best_total_val = -1.0
    best_idx = -1
    
    # Limit second-item search to top 30 by density (very low density items rarely help)
    top_k_for_second = 30
    top_second_candidates = sorted_indices[:top_k_for_second]
    
    # 2-step lookahead: for each candidate i, try top-k second items j
    for i in feasible_indices:
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        
        best_score_for_i = -1.0
        best_j_for_i = -1
        
        # Try top-k second items
        for j in top_second_candidates:
            if j == i:
                continue
            w_j = remaining_weights[j]
            if w_i + w_j > remaining_capacity:
                continue
                
            v_j = remaining_values[j]
            
            # Simulate greedy for the rest, excluding i and j
            exclude_set = {i, j}
            future_val = _simulate_greedy_value_fast(
                remaining_capacity - w_i - w_j, 
                remaining_weights, 
                remaining_values, 
                sorted_indices, 
                exclude_set
            )
            
            total_val = v_i + v_j + future_val
            if total_val > best_score_for_i:
                best_score_for_i = total_val
                best_j_for_i = j
        
        # Also consider the case where we pick i, and let greedy handle the rest
        exclude_set = {i}
        future_val = _simulate_greedy_value_fast(
            remaining_capacity - w_i, 
            remaining_weights, 
            remaining_values, 
            sorted_indices, 
            exclude_set
        )
        greedy_only_val = v_i + future_val
        
        if best_score_for_i < greedy_only_val:
            best_score_for_i = greedy_only_val
            best_j_for_i = -1  # Use greedy continuation
        
        if best_score_for_i > best_total_val:
            best_total_val = best_score_for_i
            best_idx = i
            
        # Check time
        if time.time() > deadline:
            break
    
    if best_idx == -1:
        return -1
    
    # Final polish: try to improve by swapping the last item with a better alternative
    # This is a local search step that can catch cases where greedy made a suboptimal last choice
    if time.time() < deadline - 5.0:  # Only if we have time
        # Simulate the full solution with our chosen first item
        chosen = [best_idx]
        current_cap = remaining_capacity - remaining_weights[best_idx]
        
        # Greedily fill the rest
        for idx in sorted_indices:
            if idx in chosen:
                continue
            if remaining_weights[idx] <= current_cap:
                chosen.append(idx)
                current_cap -= remaining_weights[idx]
                if current_cap <= 1e-9:
                    break
        
        # Try to swap the last few items with better alternatives
        if len(chosen) >= 2:
            # Try swapping the last item
            last_item = chosen[-1]
            last_w = remaining_weights[last_item]
            last_v = remaining_values[last_item]
            
            # Find a better alternative that fits
            for candidate in sorted_indices:
                if candidate in chosen or candidate == last_item:
                    continue
                if remaining_weights[candidate] > last_w + 1e-9:
                    continue
                if remaining_values[candidate] > last_v:
                    # Swap and check if it improves total value
                    new_chosen = chosen[:-1] + [candidate]
                    # Verify all items fit
                    total_w = sum(remaining_weights[i] for i in new_chosen)
                    if total_w <= remaining_capacity + 1e-9:
                        # This swap is valid and improves value
                        # But we can only return the first item, so this doesn't directly help
                        # However, it indicates our first choice might not be optimal
                        pass
            
            # Try swapping the second-to-last item
            if len(chosen) >= 3:
                second_last = chosen[-2]
                second_last_w = remaining_weights[second_last]
                second_last_v = remaining_values[second_last]
                
                for candidate in sorted_indices:
                    if candidate in chosen or candidate == second_last:
                        continue
                    if remaining_weights[candidate] > second_last_w + 1e-9:
                        continue
                    if remaining_values[candidate] > second_last_v:
                        new_chosen = chosen[:-2] + [candidate, chosen[-1]]
                        total_w = sum(remaining_weights[i] for i in new_chosen)
                        if total_w <= remaining_capacity + 1e-9:
                            pass
    
    return int(best_idx)
