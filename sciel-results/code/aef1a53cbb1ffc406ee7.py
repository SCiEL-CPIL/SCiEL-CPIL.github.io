import sys
import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n = 12
    points = range(n)
    
    # Generate all 5-subsets
    all_5_subsets = list(combinations(points, 5))
    subset_to_id = {s: i for i, s in enumerate(all_5_subsets)}
    num_5_subsets = len(all_5_subsets) # 792
    
    # Generate all 6-subsets
    all_6_subsets = list(combinations(points, 6))
    num_6_subsets = len(all_6_subsets) # 924
    
    # Precompute row masks
    # row_mask[r] is an integer with bits set for the 5-subsets covered by block r
    row_mask = [0] * num_6_subsets
    # col_rows[c] is a list of row indices that cover 5-subset c
    col_rows = [[] for _ in range(num_5_subsets)]
    
    for r, block in enumerate(all_6_subsets):
        mask = 0
        for i in range(6):
            sub = block[:i] + block[i+1:]
            c = subset_to_id[sub]
            mask |= (1 << c)
            col_rows[c].append(r)
        row_mask[r] = mask
        
    # Initial available mask: all bits 0..791 set
    # (1 << 792) - 1
    initial_available = (1 << num_5_subsets) - 1
    
    solution = []
    
    # To allow returning best partial solution if time runs out
    best_solution = []
    best_len = 0
    
    # Symmetry Breaking: Fix the first block to (0, 1, 2, 3, 4, 5)
    # This reduces the search space significantly without losing optimality
    # because the design is symmetric under S_12.
    fixed_block = (0, 1, 2, 3, 4, 5)
    fixed_row_idx = -1
    for i, b in enumerate(all_6_subsets):
        if b == fixed_block:
            fixed_row_idx = i
            break
            
    if fixed_row_idx == -1:
        # Fallback: no symmetry breaking
        fixed_row_idx = None
    
    def solve(available_mask, selected_rows, depth):
        nonlocal best_solution, best_len, deadline
        
        # Time check
        if time.time() > deadline:
            return False
            
        if not available_mask:
            # Found complete solution
            if len(selected_rows) > best_len:
                best_solution = selected_rows.copy()
                best_len = len(selected_rows)
            return True
            
        # Heuristic: Find column with min valid rows
        min_count = float('inf')
        best_col = -1
        best_rows = []
        
        # Iterate through available columns
        # We can iterate through all 792 cols, but we can skip those not in available_mask
        # A faster way: iterate through all cols, check if bit is set.
        # Or, maintain a list of active cols?
        # Given 792 is small, iterating all is fine.
        
        for c in range(num_5_subsets):
            if not (available_mask & (1 << c)):
                continue
                
            valid_rows = []
            for r in col_rows[c]:
                # Check if row r is compatible: all its cols are in available_mask
                # i.e., row_mask[r] is a subset of available_mask
                if (row_mask[r] & ~available_mask) == 0:
                    valid_rows.append(r)
                    
            count = len(valid_rows)
            if count == 0:
                return False # Dead end
                
            if count < min_count:
                min_count = count
                best_col = c
                best_rows = valid_rows
                
                if count == 1:
                    break
        
        # Try each row in best_rows
        for r in best_rows:
            # Select row r
            new_available = available_mask & ~row_mask[r]
            
            # Recursive call
            if solve(new_available, selected_rows + [r], depth + 1):
                return True
                
        return False

    # Start solving
    if fixed_row_idx is not None:
        # Start with the fixed block selected
        first_available = initial_available & ~row_mask[fixed_row_idx]
        solve(first_available, [fixed_row_idx], 1)
    else:
        solve(initial_available, [], 0)
    
    # If we found a solution (or partial), return it
    if best_solution:
        return [list(all_6_subsets[r]) for r in best_solution]
    else:
        return []
