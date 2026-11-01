"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses a two-phase approach:
    1. Generate candidate valid rows up to a bound
    2. Use backtracking with forward checking to select non-conflicting rows
    """
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def generate_rows_up_to(max_val, max_rows=10000):
        """Generate valid rows with max element <= max_val, limited to max_rows"""
        rows = []
        
        def bt(current, local_diffs):
            if len(rows) >= max_rows:
                return
            if len(current) == k + 1:
                rows.append(tuple(current))
                return
            
            last = current[-1]
            remaining = k + 1 - len(current)
            if last + remaining > max_val:
                return
            
            for x in range(last + 1, max_val + 1):
                new_diffs = set()
                valid = True
                for a in current:
                    d = x - a
                    if d in local_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    bt(current + [x], local_diffs | new_diffs)
                    if len(rows) >= max_rows:
                        return
        
        bt([0], set())
        return rows
    
    def select_non_conflicting(all_rows, n_needed):
        """Select n_needed rows with no shared differences using backtracking"""
        row_diffs_list = [row_diffs(r) for r in all_rows]
        indexed = [(all_rows[i], row_diffs_list[i]) for i in range(len(all_rows))]
        indexed.sort(key=lambda x: x[0][-1])
        
        used = set()
        selected = []
        
        def bt(idx, depth):
            if depth == n_needed:
                return True
            if len(indexed) - idx < n_needed - depth:
                return False
            
            for i in range(idx, len(indexed)):
                row, diffs = indexed[i]
                if diffs & used:
                    continue
                
                used.update(diffs)
                selected.append(row)
                
                if bt(i + 1, depth + 1):
                    return True
                
                selected.pop()
                used.difference_update(diffs)
            
            return False
        
        success = bt(0, 0)
        return success, selected
    
    start_time = time.time()
    timeout = 110
    
    # Try increasing M from lower_bound
    for M in range(lower_bound, lower_bound + 60):
        if time.time() - start_time > timeout:
            break
        
        all_rows = generate_rows_up_to(M, max_rows=5000)
        
        if len(all_rows) < n_rows:
            continue
        
        success, selected = select_non_conflicting(all_rows, n_rows)
        
        if success:
            return [list(r) for r in selected]
    
    # Fallback: greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
