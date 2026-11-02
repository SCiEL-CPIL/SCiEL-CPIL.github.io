<<<<<<< SEARCH
    # 4. Final quick SA pass with very low temperature
    final_deadline = deadline
    T = 0.2
    T_min = 0.01
    cool = 0.99999
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < final_deadline:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(20, old_val - min_allowed)
        
        if max_decrease <= 0:
            continue
        
        # Try a random subset of decreasing moves
        n_try = min(max_decrease, random.randint(3, 8))
        candidates = random.sample(range(1, max_decrease + 1), n_try)
        candidates = [old_val - d for d in candidates]
        
        static_pi = static_diffs[ri][pi]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
            
            old_diffs = row_diffs[ri]
            
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.05
            else:
                accept = False
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    
                moved = True
                break
    
    return best_rows
=======
    # 4. Final high-precision polish: exhaustive single-step descent on max-row
    final_deadline = deadline
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    # First: try row reconstruction on max rows
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    for ri in max_row_indices:
        target_max = row_maxes[ri]
        for attempt_max in [target_max - 1, target_max - 2, target_max - 3]:
            if attempt_max <= 0:
                break
            new_row, success = _rebuild_row(ri, current_rows, all_used, k, attempt_max)
            if success and max(new_row) < target_max:
                old_diffs = row_diffs[ri]
                new_diffs = _row_diffs_set(new_row)
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = new_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = max(new_row)
                current_val = max(row_maxes)
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                break
    
    # Then: exhaustive single-step descent
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    improved = True
    while improved and time.time() < final_deadline:
        improved = False
        for ri in max_row_indices:
            for pi in range(1, k + 1):
                old_val = current_rows[ri][pi]
                min_allowed = current_rows[ri][pi - 1] + 1
                max_decrease = min(30, old_val - min_allowed)
                if max_decrease <= 0:
                    continue
                
                static_pi = static_diffs[ri][pi]
                for new_val in range(old_val - 1, min_allowed - 1, -1):
                    if pi < k and new_val >= current_rows[ri][pi + 1]:
                        continue
                    
                    new_diffs, valid_internal = _compute_new_diffs_fast(
                        current_rows[ri], pi, new_val, static_pi)
                    if not valid_internal:
                        continue
                    
                    old_diffs = row_diffs[ri]
                    conflict = False
                    for d in new_diffs:
                        if d in all_used and d not in old_diffs:
                            conflict = True
                            break
                    if conflict:
                        continue
                    
                    if pi == k:
                        cand_row_max = new_val
                    else:
                        cand_row_max = current_rows[ri][k]
                    
                    old_row_max = row_maxes[ri]
                    row_maxes[ri] = cand_row_max
                    new_global_max = max(row_maxes)
                    row_maxes[ri] = old_row_max
                    
                    if new_global_max < current_val:
                        all_used -= old_diffs
                        all_used |= new_diffs
                        current_rows[ri][pi] = new_val
                        row_diffs[ri] = new_diffs
                        row_maxes[ri] = cand_row_max
                        current_val = new_global_max
                        static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                        
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                        
                        improved = True
                        break
        
        # Update max_row_indices
        max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    
    return best_rows
>>>>>>> REPLACE
