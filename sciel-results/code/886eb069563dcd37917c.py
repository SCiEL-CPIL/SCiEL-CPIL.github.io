    def solve(scope):
        used_mask = 0
        rows = []
        
        def backtrack(row_idx, current_row, current_max):
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row[:])
                if backtrack(row_idx + 1, [0], 0):
                    return True
                rows.pop()
                return False
            
            min_next = current_max + 1
            max_next = scope
            
            r = len(current_row)
            rem_in_row = (k + 1) - r - 1
            
            # Pre-calculate needed diffs
            needed_current_row = 0
            for i in range(1, rem_in_row + 1):
                needed_current_row += (r + i)
            needed_future_rows = (n_rows - row_idx - 1) * diffs_per_row
            total_needed_after_placement = needed_current_row + needed_future_rows
            
            total_possible = scope * (scope + 1) // 2
            
            for x in range(min_next, max_next + 1):
                if x + rem_in_row > scope:
                    break
                
                valid = True
                new_mask = 0
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if not valid:
                    continue
                
                # Check capacity
                # new_used_count = used_mask.bit_count() + r
                # If total_possible - new_used_count < total_needed_after_placement: continue
                
                # Optimization: 
                # Instead of bit_count every time, we can track it? 
                # bit_count is very fast in CPython 3.10+.
                
                if (total_possible - (used_mask.bit_count() + r)) < total_needed_after_placement:
                    continue
                
                current_row.append(x)
                
                if backtrack(row_idx, current_row, x) | new_mask: # Wait, need to pass new mask? 
                    # The recursion needs to use the updated mask.
                    # I should pass used_mask as a parameter or use nonlocal.
                    # Using nonlocal is cleaner.
                    pass
                
                current_row.pop()
            
            return False
        
        # Need to handle used_mask update in recursion.
        # Let's use a nonlocal variable or pass it.
        # Passing it is safer for backtracking.
        
        def backtrack2(row_idx, current_row, current_max, used_mask):
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row[:])
                if backtrack2(row_idx + 1, [0], 0, used_mask):
                    return True
                rows.pop()
                return False
            
            min_next = current_max + 1
            max_next = scope
            
            r = len(current_row)
            rem_in_row = (k + 1) - r - 1
            
            needed_current_row = 0
            for i in range(1, rem_in_row + 1):
                needed_current_row += (r + i)
            needed_future_rows = (n_rows - row_idx - 1) * diffs_per_row
            total_needed_after_placement = needed_current_row + needed_future_rows
            
            total_possible = scope * (scope + 1) // 2
            
            for x in range(min_next, max_next + 1):
                if x + rem_in_row > scope:
                    break
                
                valid = True
                new_mask = 0
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if not valid:
                    continue
                
                if (total_possible - (used_mask.bit_count() + r)) < total_needed_after_placement:
                    continue
                
                current_row.append(x)
                new_used_mask = used_mask | new_mask
                
                if backtrack2(row_idx, current_row, x, new_used_mask):
                    return True
                
                current_row.pop()
            
            return False
        
        return backtrack2(0, [0], 0, 0)
