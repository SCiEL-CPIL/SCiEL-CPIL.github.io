"""Improved program for difference triangle set (4,7) using backtracking with bitsets."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct over all rows.
    Minimizes the largest entry.
    
    Uses a randomized backtracking search with bitsets for efficient difference tracking.
    """
    # Parameters for the specific task (4,7)
    if n_rows != 4 or k != 7:
        # Fallback for other parameters: simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Specific solver for (4,7)
    # Target max entry: 134
    MAX_ENTRY = 134
    
    def solve_with_max_entry(max_val, timeout=100):
        """
        Attempts to construct 4 rows of 8 integers with max entry <= max_val.
        Returns a list of 4 rows if successful, else None.
        """
        start_time = time.time()
        used_diffs = [False] * (max_val + 1)
        rows = []
        
        # We will use a recursive function with a node counter to enforce timeout
        node_count = 0
        max_nodes = 1000000  # Rough limit to avoid infinite loops in bad cases
        
        def backtrack(row_idx, current_marks, last_mark):
            nonlocal node_count
            node_count += 1
            if node_count > max_nodes or time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                return True
            
            # Build the next row
            # current_marks starts with [0] if row_idx > 0, but we pass it in.
            # Actually, let's manage the row construction inside.
            
            # For row 0, we start with [0].
            # For subsequent rows, we also start with [0].
            
            # Heuristic: Try marks in a slightly randomized order?
            # No, standard backtracking is better for finding *a* solution.
            # However, to avoid getting stuck in a specific deep branch, we can
            # randomize the choice of the *next* mark among valid candidates?
            # This turns it into a randomized backtracking.
            
            def build_row(marks_so_far, last_m):
                nonlocal node_count
                if len(marks_so_far) == k + 1:
                    rows.append(marks_so_far)
                    success = backtrack(row_idx + 1, [], 0)
                    if success:
                        return True
                    else:
                        rows.pop()
                        return False
                
                # Generate candidates
                candidates = list(range(last_m + 1, max_val + 1))
                # Shuffle to randomize search path
                random.shuffle(candidates)
                
                for next_mark in candidates:
                    node_count += 1
                    if node_count > max_nodes or time.time() - start_time > timeout:
                        return False
                        
                    # Check differences
                    valid = True
                    new_diffs = []
                    for m in marks_so_far:
                        d = next_mark - m
                        if d > max_val or used_diffs[d]:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Mark differences as used
                    for d in new_diffs:
                        used_diffs[d] = True
                    
                    # Recurse
                    if build_row(marks_so_far + [next_mark], next_mark):
                        return True
                    
                    # Unmark
                    for d in new_diffs:
                        used_diffs[d] = False
                
                return False
            
            # Start building the row with [0]
            return build_row([0], 0)
        
        # Run backtracking with a few restarts if it fails
        for _ in range(5):
            if backtrack(0, [], 0):
                return list(rows)
            # Reset state for next attempt
            used_diffs = [False] * (max_val + 1)
            rows = []
            node_count = 0
            start_time = time.time()
            
        return None
    
    result = solve_with_max_entry(134)
    if result is not None:
        return result
        
    # If 134 fails, try higher
    for m in range(135, 150):
        result = solve_with_max_entry(m, timeout=10)
        if result is not None:
            return result
            
    # Fallback
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
