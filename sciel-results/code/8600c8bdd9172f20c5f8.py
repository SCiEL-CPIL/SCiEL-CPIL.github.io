def construct(n_rows, k):
    used = set()
    rows = []
    
    def build_row():
        row = [0]
        row_diffs = set()  # differences within the current row
        
        def backtrack(pos):
            if pos == k + 1:
                return True
            prev = row[-1]
            upper = prev + 120
            
            for x in range(prev + 1, upper + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used or new_diffs & row_diffs:
                    continue
                
                row.append(x)
                row_diffs.update(new_diffs)
                
                # Forward check
                feasible = True
                for rem_pos in range(pos + 1, k + 1):
                    rem_prev = row[-1]
                    has_candidate = False
                    for y in range(rem_prev + 1, rem_prev + 25):
                        y_diffs = {y - a for a in row}
                        if not (y_diffs & used) and not (y_diffs & row_diffs):
                            has_candidate = True
                            break
                    if not has_candidate:
                        feasible = False
                        break
                
                if feasible:
                    if backtrack(pos + 1):
                        used.update(row_diffs)
                        return True
                
                row.pop()
                row_diffs.difference_update(new_diffs)
            
            return False
        
        if backtrack(1):
            return row
        return None
    
    for _ in range(n_rows):
        row = build_row()
        if row is None:
            row = [0]
            x = 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used and not new & row_diffs:
                    row.append(x)
                    row_diffs.update(new)
            used.update(row_diffs)
        rows.append(row)
    
    return rows
