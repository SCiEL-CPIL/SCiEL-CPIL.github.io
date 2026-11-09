    def try_decrease(rows, row_idx, elem_idx, used_diffs_global):
        row = rows[row_idx]
        if elem_idx == 0:
            return False
        
        current_val = row[elem_idx]
        min_val = row[elem_idx - 1] + 1
        
        # Try values from min_val to current_val - 1
        # We want the smallest valid one to compress the row.
        for val in range(min_val, current_val):
            old_val = row[elem_idx]
            row[elem_idx] = val
            
            if is_valid_row(row, used_diffs_global, row_idx):
                return True
            
            row[elem_idx] = old_val
            
        return False
