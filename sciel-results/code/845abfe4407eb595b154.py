"""Improved program for difference triangle set (4,7) using backtracking with bitsets."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct over all rows.
    Minimizes the largest entry.
    """
    # Total number of differences per row: C(k+1, 2)
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound on max entry
    lower_bound = total_diffs_needed
    
    # We know best known is 134. Let's try to find a solution with M <= 140.
    # If we find one, it's likely optimal or near-optimal.
    
    def solve(max_entry):
        # Bitset for used differences. Index i corresponds to diff i.
        # We use a Python integer as a bitset.
        # Bit i is set if diff i is used.
        used_diffs = 0
        rows = []
        
        # Pre-compute mask for valid diffs [1, max_entry]
        valid_mask = (1 << (max_entry + 1)) - 2  # Bits 1 to max_entry set
        
        def backtrack(row_marks, used_diffs_local):
            """
            Recursively build a single row.
            Returns True if successful, updating rows and used_diffs via nonlocal/closure (careful with state).
            Since we need to update global state, we'll use a class or mutable containers.
            """
            if len(row_marks) == k + 1:
                # Row complete
                rows.append(row_marks)
                return True
            
            current_max = row_marks[-1]
            # Try next mark
            # Heuristic: try marks that are "spread out" or just sequential?
            # Sequential is standard.
            
            # Pruning: Check if enough diffs are available for the rest of the row
            marks_left = k + 1 - len(row_marks)
            # Diffs added by next mark: len(row_marks)
            # Diffs added by mark after that: len(row_marks) + 1
            # ...
            # Total diffs needed = sum_{i=0}^{marks_left-1} (len(row_marks) + i)
            needed_diffs = marks_left * len(row_marks) + marks_left * (marks_left - 1) // 2
            
            # Count available diffs in [1, max_entry]
            available_bits = valid_mask & ~used_diffs_local
            # Fast count of bits
            available_count = available_bits.bit_count()
            
            if available_count < needed_diffs:
                return False
            
            # Also check if we have enough "room" for marks
            # We need marks_left marks in (current_max, max_entry]
            if max_entry - current_max < marks_left:
                return False
            
            for next_mark in range(current_max + 1, max_entry + 1):
                # Calculate new diffs
                new_diffs = []
                diff_bitset = 0
                conflict = False
                for m in row_marks:
                    d = next_mark - m
                    if used_diffs_local & (1 << d):
                        conflict = True
                        break
                    new_diffs.append(d)
                    diff_bitset |= (1 << d)
                
                if conflict:
                    continue
                
                # Recursive step
                # We need to pass the updated used_diffs
                new_used = used_diffs_local | diff_bitset
                
                if backtrack(row_marks + [next_mark], new_used):
                    return True
            
            return False
        
        # Build rows one by one
        # To improve efficiency, we can randomize the order of rows? No, rows are symmetric.
        # But the order of marks within a row matters for pruning.
        
        # Try to construct all rows
        # We need to restore state if a row fails.
        # Since `used_diffs` is an int, we can just pass it.
        
        # We need to clear rows if we fail
        rows.clear()
        
        # Start with first row
        if backtrack([0], 0):
            # Check if we actually built n_rows rows
            if len(rows) == n_rows:
                return rows
        return None

    # Try increasing max_entry
    # Start from lower_bound, go up to 150
    # If we find a solution, return it.
    
    # Optimization: Use a timeout or iteration limit to avoid hanging on hard instances
    # But since we are pruning strongly, it should be fast for M=134.
    
    for M in range(lower_bound, 150):
        result = solve(M)
        if result is not None:
            return result
            
    # Fallback: Greedy construction
    # This ensures we return a valid solution even if backtracking fails
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
