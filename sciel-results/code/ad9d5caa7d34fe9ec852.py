<<<<<<< SEARCH
    def scope_squeeze(rows, time_limit):
        """Try to reduce scope by forcing max element down."""
        current_rows = [r[:] for r in rows]
        current_scope = calc_scope(current_rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        # Build used_diffs
        used_diffs = set()
        for row in current_rows:
            used_diffs |= set(row_diffs(row))

        # Try to reduce the maximum element
        while time.time() < time_limit and best_scope > 0:
            # Find the row with the maximum element
            max_val = best_scope
            target_row = -1
            target_pos = -1
            for r in range(n_rows):
                if current_rows[r][k] == max_val:
                    target_row = r
                    target_pos = k
                    break

            if target_row < 0:
                break

            # Try to reduce
            improved = False
            for new_val in range(max_val - 1, 0, -1):
                if new_val <= current_rows[target_row][k - 1]:
                    break
                # Check validity
                old_diffs = set()
                new_diffs = set()
                for i in range(k + 1):
                    if i != target_pos:
                        old_diffs.add(abs(current_rows[target_row][i] - max_val))
                        new_diffs.add(abs(current_rows[target_row][i] - new_val))

                test_row = current_rows[target_row][:]
                test_row[target_pos] = new_val
                if not is_valid_row(test_row):
                    continue

                remaining = used_diffs - old_diffs
                if new_diffs & remaining:
                    continue

                # Apply
                used_diffs -= old_diffs
                current_rows[target_row][target_pos] = new_val
                used_diffs |= new_diffs
                current_scope = new_val
                improved = True
                break

            if not improved:
                # Try SA for a short burst to escape local optimum
                if time.time() < time_limit - 2:
                    sa_rows, sa_scope = sa_optimize(current_rows, used_diffs, time.time() + 3)
                    if sa_scope < best_scope:
                        best_scope = sa_scope
                        best_rows = sa_rows
                        current_rows = [r[:] for r in sa_rows]
                        used_diffs = set()
                        for row in current_rows:
                            used_diffs |= set(row_diffs(row))
                else:
                    break

            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]

        return best_rows, best_scope
=======
    def scope_squeeze(rows, time_limit):
        """Try to reduce scope by forcing max element down."""
        current_rows = [r[:] for r in rows]
        current_scope = calc_scope(current_rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        # Build used_diffs and per-row diff sets
        used_diffs = set()
        row_diff_sets = []
        for row in current_rows:
            rd = set(row_diffs(row))
            row_diff_sets.append(rd)
            used_diffs |= rd

        # Try to reduce the maximum element iteratively
        while time.time() < time_limit and best_scope > 1:
            # Find all rows that have the current max as their last element
            max_val = best_scope
            target_rows = []
            for r in range(n_rows):
                if current_rows[r][k] == max_val:
                    target_rows.append(r)

            if not target_rows:
                # Maybe the max is in an interior position (shouldn't happen for sorted rows)
                for r in range(n_rows):
                    if max(current_rows[r]) == max_val:
                        target_rows.append(r)
                        break

            if not target_rows:
                break

            improved = False
            for tr in target_rows:
                # Try to reduce the last element of this row
                for new_val in range(max_val - 1, current_rows[tr][k-1], -1):
                    old_diffs = set()
                    new_diffs = set()
                    for i in range(k + 1):
                        if i != k:
                            old_diffs.add(abs(current_rows[tr][i] - max_val))
                            new_diffs.add(abs(current_rows[tr][i] - new_val))

                    other_diffs = row_diff_sets[tr] - old_diffs
                    if new_diffs & other_diffs:
                        continue
                    if len(new_diffs) != len(old_diffs):
                        continue

                    remaining = used_diffs - old_diffs
                    if new_diffs & remaining:
                        continue

                    # Apply
                    used_diffs -= old_diffs
                    current_rows[tr][k] = new_val
                    used_diffs |= new_diffs
                    row_diff_sets[tr] = other_diffs | new_diffs
                    current_scope = new_val
                    improved = True
                    break

                if improved:
                    break

            if not improved:
                # Try SA for a short burst to escape local optimum
                if time.time() < time_limit - 3:
                    sa_rows, sa_scope = sa_optimize(current_rows, used_diffs, time.time() + 4, T_init=5.0, cooling=0.9997, min_T=0.08)
                    if sa_scope < best_scope:
                        best_scope = sa_scope
                        best_rows = sa_rows
                        current_rows = [r[:] for r in sa_rows]
                        used_diffs = set()
                        row_diff_sets = []
                        for row in current_rows:
                            rd = set(row_diffs(row))
                            row_diff_sets.append(rd)
                            used_diffs |= rd
                    else:
                        # No improvement from SA, try one more squeeze attempt
                        if time.time() < time_limit - 1:
                            continue
                        else:
                            break
                else:
                    break

            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]

        return best_rows, best_scope
>>>>>>> REPLACE

