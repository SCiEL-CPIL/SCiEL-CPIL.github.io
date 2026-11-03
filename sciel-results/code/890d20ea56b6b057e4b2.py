"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import sys
import time

def construct(n_rows, k):
    # Difference triangle set: choose n_rows rows of k+1 increasing integers starting at 0
    # such that all positive differences a_j - a_i (i < j) are distinct across all rows.
    # Minimize the largest entry.
    
    best_rows = None
    best_max = float('inf')
    
    # Use randomized backtracking to find a compact solution
    # The problem size is small (4 rows, 8 elements), so this should be fast.
    
    start_time = time.time()
    max_time = 110.0  # Leave some time for overhead
    
    # We will run multiple randomized searches
    # Each search tries to build the rows one by one.
    
    for attempt in range(10000):
        if time.time() - start_time > max_time:
            break
            
        rows, max_val = _solve_randomized(n_rows, k, best_max)
        
        if rows is not None and max_val < best_max:
            best_max = max_val
            best_rows = rows
            
            # If we hit the theoretical lower bound (counting bound), we can stop
            # Counting bound: n_rows * (k+1)*k/2 distinct diffs.
            # Min max val is at least this number.
            # For n_rows=4, k=7: 4 * 28 = 112.
            # If best_max == 112, it's optimal.
            if best_max <= n_rows * (k + 1) * k // 2:
                break
                
    if best_rows is None:
        # Fallback to a simple greedy if no solution found (should not happen)
        best_rows = _greedy_fallback(n_rows, k)
        
    return best_rows

def _solve_randomized(n_rows, k, upper_bound):
    """
    Attempts to find a valid configuration with max value < upper_bound.
    Uses randomized backtracking.
    Returns (rows, max_val) or (None, None) if failed.
    """
    rows = []
    used_diffs = set()
    
    # Node limit for this specific attempt to prevent hanging
    node_limit = 2000
    
    def backtrack(row_idx, current_row, x_start):
        nonlocal used_diffs, rows, node_limit
        
        if node_limit <= 0:
            return None
            
        if len(current_row) == k + 1:
            # Row is complete
            if row_idx + 1 == n_rows:
                # All rows complete
                return [list(r) for r in rows + [current_row]]
            
            # Save state
            rows.append(list(current_row))
            # Recurse to next row
            result = backtrack(row_idx + 1, [0], 1)
            rows.pop()
            
            if result is not None:
                return result
            return None
        
        # Try to add next element to current_row
        # We need to pick x > current_row[-1]
        # To minimize max, we try smaller x first, but randomize order
        # We also need to ensure all new diffs x - a for a in current_row are not in used_diffs
        
        # Generate candidates
        # We can limit the search space by not going too far beyond the current max
        # But we don't know the final max yet.
        # Let's try candidates from x_start up to some limit.
        # A good limit is current_max + 50 or so, but we can increase if stuck.
        # For simplicity, let's try a window of candidates.
        
        current_max = current_row[-1]
        # Try a range of candidates
        # We want to explore different structures.
        # Let's pick a random subset of candidates or shuffle a range.
        
        # To ensure progress, we must eventually find a valid x.
        # If we restrict the range too much, we might fail.
        # Let's try candidates from x_start to x_start + 100.
        # If none work, we return None (backtrack).
        
        candidates = list(range(x_start, x_start + 100))
        random.shuffle(candidates)
        
        for x in candidates:
            node_limit -= 1
            if node_limit <= 0:
                return None
                
            # Check validity
            valid = True
            new_diffs = set()
            for a in current_row:
                d = x - a
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                # Tentatively add x
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                # Recurse
                result = backtrack(row_idx, current_row, x + 1)
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
                
                if result is not None:
                    return result
        
        return None
    
    # Start backtracking
    # We start with row 0, element [0]
    result = backtrack(0, [0], 1)
    
    if result is not None:
        max_val = max(max(row) for row in result)
        if max_val < upper_bound:
            return result, max_val
    return None, None

def _greedy_fallback(n_rows, k):
    """Simple greedy to ensure we return *something* valid."""
    rows = []
    used_diffs = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
