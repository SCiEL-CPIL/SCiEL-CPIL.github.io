<<<<<<< SEARCH
def _sa_optimize(rows, n_rows, k, deadline):
    """
    Simulated Annealing with cost = (collisions, scope).
    """
    state = SAState(rows)
    best_rows = [list(r) for r in state.rows]
    best_collisions, best_scope = state.get_cost()
    current_collisions, current_scope = state.get_cost()

    T = 100.0
    cooling = 0.999
    min_T = 0.5

    while time.time() < deadline - 3:
        # Decide move type: 70% perturb element, 30% rebuild row
        if random.random() < 0.7:
            # Perturb a single element
            r_idx = random.randint(0, n_rows - 1)
            row = state.rows[r_idx]
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]

            # Determine range
            lower = row[pos - 1] + 1
            if pos < len(row) - 1:
                upper = row[pos + 1] - 1
            else:
                # Allow expansion to escape local optima
                upper = best_scope + 20

            if lower > upper:
                continue

            # Try a few random values
            for _ in range(2):
                new_val = random.randint(lower, upper)
                if new_val == old_val:
                    continue

                old_collisions, old_scope = state.get_cost()
                if state.try_move(r_idx, pos, new_val):
                    new_collisions, new_scope = state.get_cost()
                    # Cost delta: (collisions, scope)
                    delta_c = new_collisions - old_collisions
                    delta_s = new_scope - old_scope

                    # Acceptance: prefer lower collisions, then lower scope
                    # Use a combined delta for SA
                    # If collisions decrease, always accept
                    # If collisions increase, accept with probability exp(-delta/T)
                    # If collisions same, accept scope decrease always, increase with prob
                    if delta_c < 0:
                        accept = True
                    elif delta_c == 0:
                        if delta_s <= 0:
                            accept = True
                        else:
                            accept = random.random() < math.exp(-delta_s / T)
                    else:
                        # Collisions increased; accept with probability
                        # that also accounts for scope improvement
                        combined_delta = delta_c * 100 + delta_s
                        accept = random.random() < math.exp(-combined_delta / T)

                    if accept:
                        current_collisions, current_scope = state.get_cost()
                        if (current_collisions, current_scope) < (best_collisions, best_scope):
                            best_collisions, best_scope = current_collisions, current_scope
                            best_rows = [list(r) for r in state.rows]
                    else:
                        state.undo_move(r_idx, pos, old_val)
                    break  # Only one move per iteration

        else:
            # Rebuild a row
            # Pick the row with the largest max element, or random
            if random.random() < 0.5:
                max_row_idx = max(range(n_rows), key=lambda i: state.rows[i][-1])
                r_idx = max_row_idx
            else:
                r_idx = random.randint(0, n_rows - 1)

            # Compute forbidden diffs (all diffs except this row's)
            old_row = state.rows[r_idx]
            old_diffs = _compute_diffs(old_row)

            # Temporarily remove old row's diffs to get forbidden set
            for d in old_diffs:
                state.diff_count[d] -= 1
                if state.diff_count[d] == 1:
                    state.collisions -= 1
                if state.diff_count[d] == 0:
                    del state.diff_count[d]

            forbidden = set(state.diff_count.keys())

            # Restore old row's diffs
            for d in old_diffs:
                if d in state.diff_count:
                    state.diff_count[d] += 1
                    if state.diff_count[d] == 2:
                        state.collisions += 1
                else:
                    state.diff_count[d] = 1

            old_collisions, old_scope = state.get_cost()

            # Try to rebuild with a hint: try to beat best_scope
            new_row, new_diffs = state.rebuild_row(
                r_idx, forbidden, k, best_scope, deadline - 2
            )

            if new_row is not None:
                new_collisions, new_scope = state.get_cost()
                delta_c = new_collisions - old_collisions
                delta_s = new_scope - old_scope

                if delta_c < 0:
                    accept = True
                elif delta_c == 0:
                    if delta_s <= 0:
                        accept = True
                    else:
                        accept = random.random() < math.exp(-delta_s / T)
                else:
                    combined_delta = delta_c * 100 + delta_s
                    accept = random.random() < math.exp(-combined_delta / T)

                if not accept:
                    # Undo the rebuild by restoring old row
                    # We need to undo the rebuild_row changes
                    # This is tricky; let's just revert by recomputing
                    # Actually, rebuild_row already applied the new row.
                    # To undo, we'd need to restore old_row.
                    # For simplicity, if not accepted, we revert by
                    # removing new diffs and adding old diffs.
                    for d in new_diffs:
                        state.diff_count[d] -= 1
                        if state.diff_count[d] == 1:
                            state.collisions -= 1
                        if state.diff_count[d] == 0:
                            del state.diff_count[d]
                    state.rows[r_idx] = old_row
                    for d in old_diffs:
                        if d in state.diff_count:
                            state.diff_count[d] += 1
                            if state.diff_count[d] == 2:
                                state.collisions += 1
                        else:
                            state.diff_count[d] = 1
                    if old_row[-1] > state.max_val:
                        state.max_val = old_row[-1]
                    # Recompute max_val if needed
                    if old_row[-1] == old_scope and state.max_val < old_scope:
                        state.max_val = max(max(r) for r in state.rows)
            else:
                # Failed to rebuild, state should be unchanged (rebuild_row restored)
                pass

        T *= cooling

        # If we have a valid solution, try to further optimize
        # Occasionally restart with a new greedy if stuck
        if (best_collisions, best_scope) == (0, best_scope) and T < 5.0:
            # We're in a valid state; do a quick local search to reduce scope
            # by trying to move the max element down
            pass

    # Return the best valid solution found
    if best_collisions == 0:
        return best_rows
    else:
        # If we never found a valid solution, return the best we have
        # (shouldn't happen since we start from valid)
        return [list(r) for r in state.rows]
=======
def _local_search(rows, deadline):
    """
    Local search to minimize scope (max value) while keeping solution valid.
    Uses a greedy repair strategy when collisions occur.
    """
    if not rows:
        return rows
    
    current_rows = [list(r) for r in rows]
    n_rows = len(current_rows)
    row_len = len(current_rows[0])
    
    # Precompute diffs for efficiency
    def get_diffs(row):
        return {row[j] - row[i] for i in range(row_len) for j in range(i+1, row_len)}
    
    used_diffs = set()
    for r in current_rows:
        used_diffs.update(get_diffs(r))
        
    current_scope = max(max(r) for r in current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    while time.time() < deadline:
        # Try to reduce the scope by moving the last element of the largest row down
        # Find the row with the maximum last element
        max_row_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
        row = current_rows[max_row_idx]
        
        # Try to reduce the last element
        old_last = row[-1]
        reduced = False
        
        # Try values from old_last - 1 down to row[-2] + 1
        for new_last in range(old_last - 1, row[-2], -1):
            # Check if new_last creates any new diffs that conflict with used_diffs
            # We need to remove diffs involving old_last first
            old_diffs = {old_last - row[i] for i in range(row_len - 1)}
            
            # Temporarily remove old_last diffs from global set
            for d in old_diffs:
                if d in used_diffs:
                    used_diffs.remove(d)
            
            # Check for conflicts with new_last
            new_diffs = {new_last - row[i] for i in range(row_len - 1)}
            conflict = False
            for d in new_diffs:
                if d in used_diffs:
                    conflict = True
                    break
            
            if not conflict:
                # Accept the move
                row[-1] = new_last
                for d in new_diffs:
                    used_diffs.add(d)
                
                new_scope = max(max(r) for r in current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
                
                reduced = True
                break
            else:
                # Restore old_last diffs
                for d in old_diffs:
                    used_diffs.add(d)
        
        if not reduced:
            # If we couldn't reduce the last element, try to move other elements
            # This is a simple heuristic: try to move a random non-last element down
            # and repair if needed. For now, if stuck, break to avoid infinite loop.
            # A more robust version would try to swap elements or use backtracking.
            # Given time constraints, we'll just break if we can't make progress on the max element.
            # However, we might be able to reduce scope by moving earlier elements.
            # Let's try to move the second to last element down.
            if row_len > 2:
                pos = row_len - 2
                old_val = row[pos]
                improved = False
                for new_val in range(old_val - 1, row[pos-1], -1):
                    # Check conflicts
                    # Remove diffs involving old_val
                    old_diffs_pos = {abs(old_val - row[i]) for i in range(row_len) if i != pos}
                    for d in old_diffs_pos:
                        if d in used_diffs:
                            used_diffs.remove(d)
                            
                    new_diffs_pos = {abs(new_val - row[i]) for i in range(row_len) if i != pos}
                    conflict = False
                    for d in new_diffs_pos:
                        if d in used_diffs:
                            conflict = True
                            break
                            
                    if not conflict:
                        row[pos] = new_val
                        for d in new_diffs_pos:
                            used_diffs.add(d)
                        new_scope = max(max(r) for r in current_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in current_rows]
                        improved = True
                        break
                    else:
                        for d in old_diffs_pos:
                            used_diffs.add(d)
                
                if not improved:
                    # Try other rows
                    random.shuffle(range(n_rows))
                    for r_idx in range(n_rows):
                        if r_idx == max_row_idx:
                            continue
                        r = current_rows[r_idx]
                        # Try to reduce last element of this row
                        old_last_r = r[-1]
                        for new_last_r in range(old_last_r - 1, r[-2], -1):
                            old_diffs_r = {old_last_r - r[i] for i in range(row_len - 1)}
                            for d in old_diffs_r:
                                if d in used_diffs:
                                    used_diffs.remove(d)
                            new_diffs_r = {new_last_r - r[i] for i in range(row_len - 1)}
                            conflict = False
                            for d in new_diffs_r:
                                if d in used_diffs:
                                    conflict = True
                                    break
                            if not conflict:
                                r[-1] = new_last_r
                                for d in new_diffs_r:
                                    used_diffs.add(d)
                                new_scope = max(max(rw) for rw in current_rows)
                                if new_scope < best_scope:
                                    best_scope = new_scope
                                    best_rows = [list(rw) for rw in current_rows]
                                break
                            else:
                                for d in old_diffs_r:
                                    used_diffs.add(d)
            
            # If no improvement in this iteration, we might be stuck.
            # To avoid infinite loops, we can break if we've tried several times without success.
            # For simplicity, we break.
            break
            
    return best_rows
>>>>>>> REPLACE

