    def find_row(used_set, k, limit):
        row = [0]
        row_diffs = set() # Diffs within the current row being built
        
        def backtrack(current_row, last_val, current_diffs):
            if len(current_row) == k + 1:
                return current_row[:]
            
            for next_val in range(last_val + 1, limit + 1):
                new = {next_val - a for a in current_row}
                
                # Check against global used
                if new & used_set:
                    continue
                    
                # Check against internal diffs
                if new & current_diffs:
                    continue
                
                # Also check if new has duplicates? 
                # new is a set, so if len(new) < len(current_row), there are duplicates.
                if len(new) < len(current_row):
                    continue

                current_row.append(next_val)
                res = backtrack(current_row, next_val, current_diffs | new)
                if res is not None:
                    return res
                current_row.pop()
            return None
            
        return backtrack(row, 0, row_diffs)
