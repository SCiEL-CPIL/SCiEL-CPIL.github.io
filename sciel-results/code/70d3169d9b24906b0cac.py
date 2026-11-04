            p = len(current_row)
            diffs_so_far_row = p * (p - 1) // 2
            total_diffs_row = k * (k + 1) // 2
            remaining_diffs_row = total_diffs_row - diffs_so_far_row
            
            remaining_rows = n_rows - current_row_idx
            # Note: current_row is one of the remaining_rows
            # So total remaining diffs = remaining_diffs_row + (remaining_rows - 1) * total_diffs_row
            
            total_diffs_needed_remaining = remaining_diffs_row + (remaining_rows - 1) * total_diffs_row
