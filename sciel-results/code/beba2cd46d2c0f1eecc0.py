import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Build n_rows rows of k+1 increasing integers starting at 0,
    such that all positive within-row differences are globally distinct.
    Minimize the largest entry.
    """
    used_diffs = set()
    rows = []
    
    # Helper to run DFS for a single row with a given max limit
    def find_best_row(used_diffs, k, max_limit):
        """
        Finds a valid row of length k+1 starting at 0 with max value <= max_limit.
        Returns the row with the smallest possible max value, or None if not found.
        """
        best_row = None
        best_max_val = max_limit + 1
        
        current_row = [0]
        
        def dfs(row, idx, max_val_so_far):
            nonlocal best_row, best_max_val
            
            if idx == k:
                if max_val_so_far < best_max_val:
                    best_max_val = max_val_so_far
                    best_row = row[:]
                return
            
            # Prune if current max is already >= best found
            if max_val_so_far >= best_max_val:
                return
                
            last = row[-1]
            # We only care about x such that x <= max_limit
            # Also, we can prune if even the smallest possible future additions
            # would exceed max_limit? Not strictly necessary for small k.
            
            for x in range(last + 1, max_limit + 1):
                # Check differences
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    dfs(row, idx + 1, max(max_val_so_far, x))
                    row.pop()
                    
                    # Early exit if we found a solution with max 1? (Impossible for k>0)
                    # Or if we found a solution that is "good enough"?
                    # We want the minimum, so we continue.
                    # However, if best_max_val becomes small, pruning is effective.
                    
        dfs(current_row, 1, 0)
        
        return best_row

    # 1. Greedy Phase to get initial rows and bounds
    for r in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs) and len(new_diffs) == len(row):
                    break
                x += 1
            row.append(x)
        rows.append(row)
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    # 2. Refinement Phase: Try to reduce the span of each row
    # We process rows in order. Reducing an earlier row might allow later rows to be smaller?
    # Actually, reducing an earlier row frees up small differences, which might help later rows.
    # So we should iterate and try to improve each row.
    
    # We'll do a few passes
    for _pass in range(5):
        improved_global = False
        for r in range(n_rows):
            current_row = rows[r]
            current_max = max(current_row)
            
            # Remove current row's diffs
            old_diffs = set()
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    old_diffs.add(current_row[j] - current_row[i])
            
            for d in old_diffs:
                used_diffs.discard(d)
                
            # Try to find a row with max < current_max
            # We can set the limit to current_max - 1
            if current_max > 1:
                new_row = find_best_row(used_diffs, k, current_max - 1)
                
                if new_row is not None:
                    new_max = max(new_row)
                    if new_max < current_max:
                        # Accept the new row
                        rows[r] = new_row
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                used_diffs.add(new_row[j] - new_row[i])
                        improved_global = True
                    else:
                        # Restore old diffs
                        for d in old_diffs:
                            used_diffs.add(d)
                else:
                    # Restore old diffs
                    for d in old_diffs:
                        used_diffs.add(d)
            else:
                for d in old_diffs:
                    used_diffs.add(d)
                
        if not improved_global:
            break
            
    return rows
