import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    # Filter to feasible items only
    feasible = []
    for i in range(n):
        w = remaining_weights[i]
        v = remaining_values[i]
        if w > 0 and w <= remaining_capacity:
            feasible.append((v, w, i))
        elif w == 0 and v > 0:
            return i  # Free value, take it
    
    if not feasible:
        return -1
    
    # Sort by density descending for greedy baseline
    feasible.sort(key=lambda x: x[0] / x[1], reverse=True)
    
    # Greedy fill to get a baseline
    greedy_val = 0
    greedy_items = []
    rem_cap = remaining_capacity
    for v, w, idx in feasible:
        if w <= rem_cap:
            greedy_val += v
            greedy_items.append(idx)
            rem_cap -= w
    
    # Try each item as the first pick, then greedy fill the rest
    best_first = feasible[0][2]
    best_total = greedy_val
    
    # Limit the number of candidates to try for speed
    candidates = feasible[:min(50, len(feasible))]
    
    for v_first, w_first, idx_first in candidates:
        if w_first > remaining_capacity:
            continue
        rem_cap_after = remaining_capacity - w_first
        # Greedy fill with remaining items
        total = v_first
        for v, w, idx in feasible:
            if idx == idx_first:
                continue
            if w <= rem_cap_after:
                total += v
                rem_cap_after -= w
        if total > best_total:
            best_total = total
            best_first = idx_first
    
    # If we still have time, try a more sophisticated approach
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        # Try local search: swap out items in the greedy solution
        # Build the greedy solution set
        greedy_set = set(greedy_items)
        current_solution = list(greedy_items)
        current_value = greedy_val
        
        # Try swapping one item out and one item in
        improved = True
        max_swaps = 100
        swap_count = 0
        
        while improved and swap_count < max_swaps:
            improved = False
            swap_count += 1
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - SAFETY_MARGIN:
                break
            
            # Try removing each item in the current solution
            for i, idx_out in enumerate(current_solution):
                if idx_out not in greedy_set:
                    continue
                # Find weight and value of this item
                w_out = 0
                v_out = 0
                for v, w, idx in feasible:
                    if idx == idx_out:
                        w_out = w
                        v_out = v
                        break
                
                new_cap = remaining_capacity - sum(
                    w for v, w, idx in feasible if idx in greedy_set and idx != idx_out
                )
                
                # Try adding items not in the solution
                for v_in, w_in, idx_in in feasible:
                    if idx_in in greedy_set:
                        continue
                    if w_in <= new_cap:
                        new_value = current_value - v_out + v_in
                        if new_value > current_value + 1e-9:
                            # Perform the swap
                            current_solution.remove(idx_out)
                            current_solution.append(idx_in)
                            greedy_set.remove(idx_out)
                            greedy_set.add(idx_in)
                            current_value = new_value
                            improved = True
                            break
        
        # Final polish: try 2-swap moves (remove two, add two)
        improved = True
        max_2swaps = 20
        two_swap_count = 0
        while improved and two_swap_count < max_2swaps:
            improved = False
            two_swap_count += 1
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - SAFETY_MARGIN:
                break
            
            sol_set = set(current_solution)
            total_w = sum(w for v, w, idx in feasible if idx in sol_set)
            
            # Try all pairs to remove
            for i_out1 in range(len(current_solution)):
                for i_out2 in range(i_out1 + 1, len(current_solution)):
                    idx1 = current_solution[i_out1]
                    idx2 = current_solution[i_out2]
                    w1, v1, w2, v2 = 0, 0, 0, 0
                    for v, w, idx in feasible:
                        if idx == idx1:
                            w1, v1 = w, v
                        elif idx == idx2:
                            w2, v2 = w, v
                    freed_cap = w1 + w2
                    freed_val = v1 + v2
                    # Try adding two items not in solution
                    for v_a, w_a, idx_a in feasible:
                        if idx_a in sol_set:
                            continue
                        if w_a > freed_cap:
                            continue
                        for v_b, w_b, idx_b in feasible:
                            if idx_b in sol_set or idx_b == idx_a:
                                continue
                            if w_a + w_b > freed_cap:
                                continue
                            new_val = current_value - freed_val + v_a + v_b
                            if new_val > current_value + 1e-9:
                                current_solution = [x for x in current_solution if x not in (idx1, idx2)]
                                current_solution.extend([idx_a, idx_b])
                                current_value = new_val
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Update best if local search improved
        if current_value > best_total:
            # Find which item to return first
            # Return the item with highest density in the improved solution
            best_density = -1
            for idx in current_solution:
                for v, w, i in feasible:
                    if i == idx:
                        density = v / w
                        if density > best_density:
                            best_density = density
                            best_first = idx
                        break
            best_total = current_value
    
    # Try DP-based approach if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        # Scale weights to integers for DP
        SCALE = 100
        cap_scaled = int(round(remaining_capacity * SCALE))
        
        if cap_scaled > 0 and cap_scaled < 5000:  # Only if DP is feasible
            # Use 1D DP with item tracking
            # dp[w] = max value achievable with weight <= w
            dp = [0.0] * (cap_scaled + 1)
            # track[i][w] = item index that was last added to reach state w, or -1
            # Use a list of lists for backtracking
            n_feasible = len(feasible)
            track = [[-1] * (cap_scaled + 1) for _ in range(n_feasible)]
            
            # Sort feasible items by index for consistent ordering
            for i, (v, w, idx) in enumerate(feasible):
                w_scaled = int(round(w * SCALE))
                if w_scaled > cap_scaled:
                    continue
                # Process in reverse to avoid using item twice
                for cw in range(cap_scaled, w_scaled - 1, -1):
                    if dp[cw - w_scaled] + v > dp[cw] + 1e-9:
                        dp[cw] = dp[cw - w_scaled] + v
                        track[i][cw] = idx
            
            # Reconstruct the optimal solution
            optimal_items = set()
            cw = cap_scaled
            for i in range(n_feasible - 1, -1, -1):
                if track[i][cw] != -1:
                    optimal_items.add(track[i][cw])
                    # Find the weight of this item
                    for v, w, idx in feasible:
                        if idx == track[i][cw]:
                            cw -= int(round(w * SCALE))
                            break
            
            # If we found a better solution via DP, update best
            dp_total = dp[cap_scaled]
            if dp_total > best_total + 1e-9:
                best_total = dp_total
                # Return the first item from the optimal solution
                # Prefer higher density items first
                best_density = -1
                for idx in optimal_items:
                    for v, w, i in feasible:
                        if i == idx:
                            density = v / w if w > 0 else float('inf')
                            if density > best_density:
                                best_density = density
                                best_first = idx
                            break
            
            # Also try: for each item in optimal solution, verify it's a good first pick
            # by checking if the remaining capacity can be filled well
            if optimal_items:
                for idx_check in list(optimal_items)[:10]:
                    for v, w, idx in feasible:
                        if idx == idx_check:
                            rem_after = remaining_capacity - w
                            # Quick greedy fill for remaining
                            fill_val = v
                            rem = rem_after
                            for v2, w2, idx2 in feasible:
                                if idx2 == idx:
                                    continue
                                if w2 <= rem:
                                    fill_val += v2
                                    rem -= w2
                            if fill_val > best_total:
                                best_total = fill_val
                                best_first = idx_check
                            break
    
    return best_first
