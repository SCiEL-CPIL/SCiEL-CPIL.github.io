<<<<<<< SEARCH
    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        # Limit the number of full passes to avoid spending too much time here
        # if we are in an ILS loop
        max_passes = 6
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Sort indices by descending max entry to target the largest rows first
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=80)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows
=======
    def local_polish(rows, final_pass=False):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        # Limit the number of full passes to avoid spending too much time here
        # if we are in an ILS loop
        max_passes = 10 if final_pass else 6
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Sort indices by descending max entry to target the largest rows first
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=80)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def high_precision_polish(rows):
        """Final polish: try to reduce each row by 1 using backtracking."""
        rows = [list(r) for r in rows]
        # Try to reduce the scope by 1
        target_scope = scope(rows)
        for reduction in range(1, 15):
            if time_left() < 10:
                break
            target = target_scope - reduction
            if target < 0:
                break
            
            # Try to rebuild the row with the largest max entry to fit within target
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                if max(rows[i]) <= target:
                    continue
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row = backtracking_row(used_other, target, timeout=2.0)
                if new_row is not None:
                    rows[i] = new_row
                    # Check if we can reduce further
                    break
            if scope(rows) < target_scope - reduction + 1:
                target_scope = scope(rows)
        return rows
>>>>>>> REPLACE

