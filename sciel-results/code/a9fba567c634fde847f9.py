def _polish_descent(rows, k, n_rows, deadline):
    current_rows = [r[:] for r in rows]
    current_scope = _compute_scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    row_maxes = [max(r) for r in current_rows]
    row_diffs = [_row_diffs_bitset(r) for r in current_rows]
    all_used = _all_used_bitset(current_rows)
    
    no_progress = 0
    max_no_progress = 5000
    
    while time.time() < deadline and no_progress < max_no_progress:
        moved_in_pass = False
        for ri in range(n_rows):
            for pi in range(1, k + 1):
                old_val = current_rows[ri][pi]
                min_allowed = current_rows[ri][pi - 1] + 1
                max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else old_val
                
                # Try to decrease
                for new_val in range(old_val - 1, min_allowed - 1, -1):
                    success, new_row_diffs, new_all_used, new_row_max = _try_move_bitset(
                        current_rows, ri, pi, new_val, all_used, row_diffs, k)
                    if success:
                        current_rows[ri][pi] = new_val
                        row_diffs[ri] = new_row_diffs
                        all_used = new_all_used
                        row_maxes[ri] = new_row_max
                        current_scope = _compute_scope(current_rows)
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in current_rows]
                            no_progress = 0
                        moved_in_pass = True
                        break
                if moved_in_pass:
                    break
            if moved_in_pass:
                break
        
        if not moved_in_pass:
            no_progress += 1
            # If stuck, do a random shake
            if no_progress % 100 == 0:
                # Pick a random row and position and jump it
                ri = random.randrange(n_rows)
                pi = random.randrange(1, k + 1)
                old_val = current_rows[ri][pi]
                min_allowed = current_rows[ri][pi - 1] + 1
                max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else old_val + 20
                if min_allowed <= max_allowed:
                    new_val = random.randint(min_allowed, max_allowed)
                    success, new_row_diffs, new_all_used, new_row_max = _try_move_bitset(
                        current_rows, ri, pi, new_val, all_used, row_diffs, k)
                    if success:
                        current_rows[ri][pi] = new_val
                        row_diffs[ri] = new_row_diffs
                        all_used = new_all_used
                        row_maxes[ri] = new_row_max
                        current_scope = _compute_scope(current_rows)
                        # Don't update best if it's worse, but keep searching
