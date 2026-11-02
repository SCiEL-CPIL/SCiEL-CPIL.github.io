            # Forward Check
            if row_idx < n_rows - 1:
                # Check if next row can start and has enough candidates
                # Candidates for next row's non-zero elements must be in [1, M]
                # and their values (which are diffs from 0) must not be in used_diffs.
                # Also, they must be distinct.
                # Simple check: count of available values in [1, M]
                avail_count = 0
                for v in range(1, m + 1):
                    if v not in used_diffs:
                        avail_count += 1
                        if avail_count >= k:
                            break
                if avail_count < k:
                    current_row.pop()
                    for d in new_diffs:
                        used_diffs.discard(d)
                    continue
