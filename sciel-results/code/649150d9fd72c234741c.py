import sys

# Increase recursion limit for safety
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    # k is such that len(row) = k+1
    L = k + 1
    if L <= 1:
        return [[0] for _ in range(n_rows)]
        
    diffs_per_row = L * (L - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound for M
    # We need total_diffs distinct values in [1, M]
    # So M >= total_diffs
    start_M = total_diffs
    
    # Upper bound
    # Best known for 4 rows, 8 elements (k=7) is 134.
    # Let's try up to 140 to be safe, or a bit higher if needed.
    # If the problem is harder, we might need more.
    # But 120s is a lot of time.
    end_M = 150
    
    # If k is different, adjust.
    # For general k, the bound might be higher.
    # But the task is specific.
    
    for M in range(start_M, end_M + 1):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, pos_idx, current_row):
            if row_idx == n_rows:
                return True
            
            if pos_idx == L:
                rows.append(current_row)
                return backtrack(row_idx + 1, 0, [0])
            
            start_val = current_row[-1] + 1
            
            # Pruning: Check if enough diffs remain
            rem_rows = n_rows - row_idx - 1
            rem_diffs_future_rows = rem_rows * diffs_per_row
            
            # Diffs to be created by the rest of the current row (after the one we place now)
            # Current element will be at index pos_idx.
            # It creates pos_idx diffs.
            # Next elements at pos_idx+1 ... L-1 create pos_idx+1 ... L-1 diffs.
            rem_curr_diffs = sum(range(pos_idx + 1, L))
            
            # We iterate x
            # We can also bound x:
            # The largest element in the row is at most M.
            # So x <= M.
            
            for x in range(start_val, M + 1):
                # Check validity of x
                valid = True
                new_diffs = []
                for p in current_row:
                    d = x - p
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Pruning check
                # len(used_diffs) + len(new_diffs) + rem_curr_diffs + rem_diffs_future_rows <= M
                if len(used_diffs) + len(new_diffs) + rem_curr_diffs + rem_diffs_future_rows > M:
                    continue
                
                # Commit
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, pos_idx + 1, current_row + [x]):
                    return True
                
                # Undo
                used_diffs.difference_update(new_diffs)
            
            return False
        
        if backtrack(0, 1, [0]):
            return rows

    # Fallback: Greedy construction if backtracking fails or times out (though it shouldn't loop forever)
    # But if no M in range works, we return the last attempted M's best effort?
    # Or just run greedy.
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < L:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
