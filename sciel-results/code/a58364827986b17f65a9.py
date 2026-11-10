"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    TIME_LIMIT = 55  # seconds, leave buffer
    
    # Split vertices into Small [2, 128] and Large [129, 256]
    # Any two distinct Large vertices are compatible (lcm > 256)
    # So we decide on Small set, and include all compatible Large vertices.
    
    small_vals = list(range(2, 129)) # 2..128
    large_vals = list(range(129, 257)) # 129..256
    
    n_small = len(small_vals)
    n_large = len(large_vals)
    
    # Weights
    w_small = [1.0 / v for v in small_vals]
    w_large = [1.0 / v for v in large_vals]
    total_large_weight = sum(w_large)
    
    # Conflicts in Small
    # conflict_small[i][j] = True if lcm(small_vals[i], small_vals[j]) <= 256
    conflict_small = [[False] * n_small for _ in range(n_small)]
    for i in range(n_small):
        vi = small_vals[i]
        for j in range(i + 1, n_small):
            vj = small_vals[j]
            if lcm(vi, vj) <= 256:
                conflict_small[i][j] = True
                conflict_small[j][i] = True
    
    # Conflicts Small -> Large
    # c_large[i] = set of indices in large_vals that conflict with small_vals[i]
    c_large = [set() for _ in range(n_small)]
    for i, v in enumerate(small_vals):
        for j, u in enumerate(large_vals):
            if lcm(v, u) <= 256:
                c_large[i].add(j)
    
    # Order small vertices by weight descending (value ascending)
    order = sorted(range(n_small), key=lambda i: w_small[i], reverse=True)
    
    best_score = 0.0
    best_state = (set(), set()) # (selected_small_indices, blocked_large_indices)
    
    def dfs(selected, blocked_large, current_small_sum, current_large_sum, next_pos):
        nonlocal best_score, best_state
        nonlocal start_time
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        # Current Score
        score = current_small_sum + current_large_sum
        
        if score > best_score:
            best_score = score
            best_state = (set(selected), set(blocked_large))
        
        # Upper Bound
        # Max additional small weight from compatible remaining vertices
        ub_add = 0.0
        for pos in range(next_pos, n_small):
            idx = order[pos]
            # Check if idx is compatible with all in selected
            can_add = True
            for s in selected:
                if conflict_small[s][idx]:
                    can_add = False
                    break
            if can_add:
                ub_add += w_small[idx]
        
        if score + ub_add <= best_score:
            return
        
        # Branch
        for pos in range(next_pos, n_small):
            idx = order[pos]
            # Check compatibility with selected
            can_add = True
            for s in selected:
                if conflict_small[s][idx]:
                    can_add = False
                    break
            
            if can_add:
                # Include idx
                new_selected = selected | {idx}
                # Update blocked_large
                new_blocked_large = blocked_large.copy()
                delta_large_weight = 0.0
                for j in c_large[idx]:
                    if j not in new_blocked_large:
                        new_blocked_large.add(j)
                        delta_large_weight += w_large[j]
                
                new_small_sum = current_small_sum + w_small[idx]
                new_large_sum = current_large_sum - delta_large_weight
                
                dfs(new_selected, new_blocked_large, new_small_sum, new_large_sum, pos + 1)
                
                if time.time() - start_time > TIME_LIMIT:
                    return
            else:
                continue
    
    # Initial call
    dfs(set(), set(), 0.0, total_large_weight, 0)
    
    # Reconstruct result
    if not best_state[0] and not best_state[1]:
        # Fallback if no solution found (should not happen)
        return [2]
    
    sel_small, blk_large = best_state
    result_small = [small_vals[i] for i in sel_small]
    result_large = [large_vals[j] for j in range(n_large) if j not in blk_large]
    
    return sorted(result_small + result_large)
# EVOLVE-BLOCK-END
