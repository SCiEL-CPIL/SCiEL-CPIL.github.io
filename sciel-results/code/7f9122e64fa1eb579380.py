"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within each row are globally distinct.
    Minimize the largest entry (the "scope").
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n = n_rows  # 4
    kk = k      # 8, so each row has kk+1 = 9 elements
    row_len = kk + 1  # 9
    
    # Get initial greedy solution
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    best_rows = greedy_construct()
    best_max = max(max(r) for r in best_rows)
    
    # Try to improve with backtracking search
    def search_with_bound(max_bound):
        """
        Try to find a valid construction with all entries <= max_bound.
        Returns the list of rows if successful, None otherwise.
        """
        if time.time() > deadline:
            return None
        
        rows = []
        used_diffs = set()
        
        def build_row(row_idx):
            if time.time() > deadline:
                return False
            
            if row_idx == n:
                return True
            
            row = [0]
            
            def extend():
                if time.time() > deadline:
                    return False
                
                if len(row) == row_len:
                    rows.append(list(row))
                    result = build_row(row_idx + 1)
                    if result:
                        return True
                    rows.pop()
                    return False
                
                # Try all possible next values
                start_val = row[-1] + 1
                
                # Collect candidates
                candidates = []
                for val in range(start_val, max_bound + 1):
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = val - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        candidates.append((val, new_diffs))
                
                # Sort by value (prefer smaller values to keep max small)
                candidates.sort(key=lambda x: x[0])
                
                for val, new_diffs in candidates:
                    row.append(val)
                    used_diffs.update(new_diffs)
                    
                    if extend():
                        return True
                    
                    used_diffs.difference_update(new_diffs)
                    row.pop()
                
                return False
            
            return extend()
        
        return build_row(0)
    
    # Try to find a solution with smaller max
    # Start from best_max - 1 and go down
    for bound in range(best_max - 1, max(best_max - 20, 100), -1):
        if time.time() > deadline:
            break
        
        result = search_with_bound(bound)
        if result is not None:
            best_rows = result
            best_max = bound
            break
    
    return best_rows

# EVOLVE-BLOCK-END
