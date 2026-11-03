            rem_in_row = diffs_per_row - (pos * (pos - 1) // 2)
            rem_in_future = (n_rows - row_idx - 1) * diffs_per_row
            total_needed = rem_in_row + rem_in_future
            
            avail_free = M - used_count - len(current_diffs)
            if avail_free < total_needed:
                return False
