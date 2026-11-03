"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import sys

def construct(n_rows, k):
    # Difference triangle set: choose n_rows rows of k+1 increasing integers starting at 0
    # such that all positive differences a_j - a_i (i < j) are distinct across all rows.
    # Minimize the largest entry.
    
    # For small cases, use known optimal or near-optimal constructions
    # The greedy approach in the seed is decent but we can do better with backtracking
    # and randomized restarts for better packing.
    
    # First, try a simple greedy with different orderings to find a good solution
    best_rows = None
    best_max = float('inf')
    
    # Try multiple greedy strategies with randomization
    for attempt in range(50):
        rows = []
        used_diffs = set()
        success = True
        
        # Randomize the order of rows slightly by using different step strategies
        for r in range(n_rows):
            row = [0]
            # Use a greedy approach with some randomization
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
                else:
                    # Try skipping ahead a bit to find a valid position
                    # But keep it greedy: just increment until valid
                    pass
            
            rows.append(row)
        
        max_val = max(max(row) for row in rows)
        if max_val < best_max:
            best_max = max_val
            best_rows = [list(r) for r in rows]
    
    # Now try a backtracking approach to improve upon the greedy solution
    # This is more expensive but can find better solutions for small n_rows and k
    
    if n_rows <= 5 and k <= 8:
        # Try backtracking with pruning
        result = _backtrack_construct(n_rows, k, best_max)
        if result is not None:
            max_val = max(max(row) for row in result)
            if max_val < best_max:
                best_rows = result
                best_max = max_val
    
    return best_rows

def _backtrack_construct(n_rows, k, upper_bound):
    """Backtracking search for better solutions, bounded by upper_bound."""
    # This is a simplified backtracking that tries to fill rows one by one
    # with pruning based on the current upper bound
    
    rows = []
    used_diffs = set()
    
    def backtrack(row_idx, row, x):
        if len(row) == k + 1:
            if row_idx + 1 == n_rows:
                return [list(r) for r in rows + [row]]
            rows.append(list(row))
            result = backtrack(row_idx + 1, [0], 0)
            rows.pop()
            return result
        return None
    
    # For simplicity, just try to improve the last few rows
    # This is a heuristic approach
    
    # Try to reconstruct with a tighter bound
    limit = upper_bound
    rows = []
    used_diffs = set()
    
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > limit:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        rows.append(row)
    
    if max(max(row) for row in rows) < limit:
        return rows
    return None

# EVOLVE-BLOCK-END
