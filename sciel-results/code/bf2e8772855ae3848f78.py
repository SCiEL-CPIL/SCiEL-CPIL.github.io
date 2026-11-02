"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
import time
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct across
    all rows. Minimize the largest entry.
    
    Uses a Constraint Satisfaction Problem (CSP) solver with backtracking and
    forward checking. It iterates the maximum allowed value M from the theoretical
    lower bound upwards, searching for a valid configuration.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on max entry M
    # We need total_diffs distinct differences.
    # The differences are integers. The smallest possible set of distinct differences
    # that can be formed by a set of size k+1 starting at 0 and ending at M is constrained.
    # A simple bound: The differences must be distinct. The sum of the smallest total_diffs
    # positive integers is total_diffs*(total_diffs+1)/2.
    # This doesn't directly give M.
    # However, for a single row of size k+1 in [0, M], the max number of distinct differences is M.
    # So n_rows * M >= total_diffs => M >= total_diffs / n_rows.
    lower_bound = (total_diffs + n_rows - 1) // n_rows
    
    # We also know M must be at least k (since we need k+1 distinct integers starting at 0).
    lower_bound = max(lower_bound, k)
    
    # Upper bound for search
    # Based on the task, best known is 112 for (7,5).
    # Let's set a generous upper bound or use time limit.
    upper_bound = 200 
    
    start_time = time.time()
    time_limit = 110  # seconds
    
    # Iterate M from lower_bound upwards
    for M in range(lower_bound, upper_bound + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Check if a solution exists with max entry <= M
        solution = solve_csp(n_rows, k, M)
        if solution is not None:
            return solution
            
    # If no solution found within bounds or time, return a greedy fallback
    # to ensure we return something valid (though possibly suboptimal)
    return greedy_fallback(n_rows, k)

def solve_csp(n_rows, k, M):
    """
    Backtracking solver to find n_rows rows of k+1 integers in [0, M]
    starting with 0, strictly increasing, such that all differences are distinct.
    """
    used_diffs = set()
    rows = []
    
    # Heuristic: Solve rows one by one.
    # To improve performance, we can try to fill rows with smaller 'gaps' first?
    # Or just standard backtracking.
    
    def backtrack(row_idx):
        if row_idx == n_rows:
            return True
        
        # Try to construct row row_idx
        # We need to find a list of k+1 integers [0, v1, ..., vk] with 0 < v1 < ... < vk <= M
        # such that all differences are in used_diffs? No, NOT in used_diffs.
        
        # We can use a recursive helper to fill the current row
        def fill_row(current_row, pos, start_val):
            if pos == k + 1:
                # Row complete, check if all diffs are valid (already checked during fill)
                # Add row to global state
                rows.append(current_row)
                # Recurse to next row
                if backtrack(row_idx + 1):
                    return True
                # Backtrack
                rows.pop()
                return False
            
            # Try next value for current_row[pos]
            # It must be > current_row[-1] (or 0 if pos==0, but pos 0 is fixed to 0)
            # Since current_row[0] is 0, we start pos from 1.
            
            # Optimization: We can pre-calculate which values are valid.
            # For a candidate x, all differences x - a for a in current_row must be in used_diffs? No, must NOT be in used_diffs.
            # Also, differences between x and future elements will be checked later.
            # But differences between x and current_row elements must be unique globally.
            
            # Let's iterate x from start_val to M
            for x in range(start_val, M + 1):
                # Check if x can be added
                # 1. x > current_row[-1] (ensured by range)
                # 2. All diffs x - a for a in current_row are not in used_diffs
                # 3. Also, we must ensure that the remaining k+1-pos-1 elements can be placed.
                #    This is hard to check cheaply.
                
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                    
                # Tentatively add x
                current_row.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                    
                # Recurse to fill next position in this row
                if fill_row(current_row, pos + 1, x + 1):
                    return True
                    
                # Backtrack
                for d in new_diffs:
                    used_diffs.remove(d)
                current_row.pop()
                
            return False
        
        # Start filling the row with [0]
        initial_row = [0]
        # The next value must be at least 1
        if fill_row(initial_row, 1, 1):
            return True
            
        return False
        
    # Call the main backtracking
    if backtrack(0):
        return rows
    return None

def greedy_fallback(n_rows, k):
    """
    Simple greedy fallback to ensure a valid solution is returned
    even if the CSP solver times out or fails to find one within bounds.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                x += 1
            else:
                x += 1
        rows.append(row)
    return rows
# EVOLVE-BLOCK-END
