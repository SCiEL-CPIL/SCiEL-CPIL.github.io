"""Rewritten program for difference triangle set (7,5)."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry."""
    
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # We'll use backtracking with iterative deepening on max value
    # Start with a greedy solution as initial bound
    best_rows = _greedy(n_rows, k)
    best_max = max(max(row) for row in best_rows)
    
    # Try to improve with backtracking, using iterative deepening
    # on the maximum allowed value
    # The counting bound suggests ~105, best known is 112
    # Try from a low bound upward
    
    # Lower bound: we need n_rows * k*(k+1)/2 distinct differences
    # Each difference is at most max_val, so we need max_val >= n_rows*k*(k+1)/2
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs  # very rough lower bound
    
    # Try iterative deepening from a reasonable start
    # Start near the greedy result and try to go lower
    # Actually, let's try from lower_bound upward, but cap iterations
    
    # First, let's try backtracking with a max bound starting from best_max-1 downward
    # to find a better solution, then try to find even better
    
    # Strategy: try max_val from lower_bound to best_max-1
    # For each, try to find a valid configuration
    
    # But this could be very slow. Let's be smarter:
    # Use backtracking with a fixed max_val and see if we can fit all rows
    
    # Let's try values from lower_bound upward
    # But we want to find the minimum, so try from low to high
    # However, the search space is huge, so we need good pruning
    
    # Alternative: use the greedy solution and try to improve it locally
    
    # Let me try a hybrid: backtracking with good heuristics
    
    # Try to find a solution with max_val = best_max - 1, best_max - 2, etc.
    # But that's expensive. Instead, let's do a single backtracking search
    # that tries to minimize max_val naturally by trying small values first.
    
    # Backtracking approach:
    # - Build rows one by one
    # - Within each row, build left to right
    # - Try candidate values in increasing order
    # - Use forward checking
    
    result = _backtrack_solve(n_rows, k, deadline)
    if result is not None:
        rmax = max(max(row) for row in result)
        if rmax < best_max:
            best_rows = result
            best_max = rmax
    
    return best_rows


def _greedy(n_rows, k):
    """Simple greedy construction."""
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


def _backtrack_solve(n_rows, k, deadline):
    """Backtracking search to find a valid configuration minimizing max value."""
    
    used_diffs = set()
    rows = []
    
    # We'll try to build rows with values as small as possible
    # Use a global best to track the minimum max value found
    
    best_max = float('inf')
    best_solution = None
    
    def backtrack(row_idx, pos, current_row):
        nonlocal best_max, best_solution, used_diffs, rows
        
        if time.time() > deadline:
            return False
        
        # If we've built all rows, update best
        if row_idx == n_rows:
            current_max = max(max(row) for row in rows)
            if current_max < best_max:
                best_max = current_max
                best_solution = [list(r) for r in rows]
            return True  # Found a solution, continue searching for better
        
        # If we're building a new row
        if pos == 0:
            current_row = [0]
            pos = 1
        
        # If row is complete, move to next row
        if pos == k + 1:
            rows.append(current_row)
            if backtrack(row_idx + 1, 0, []):
                rows.pop()
                return True
            rows.pop()
            return False
        
        # Try candidate values for current position
        # Start from last value + 1, go up
        start = current_row[-1] + 1
        
        # Upper bound: we want to keep max small
        # Try values up to some limit
        # The current best_max gives us an upper bound
        limit = best_max if best_max < float('inf') else 200
        
        for val in range(start, limit + 1):
            if time.time() > deadline:
                return False
            
            # Check if adding val creates any conflict
            new_diffs = {val - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            
            # Try this value
            current_row.append(val)
            used_diffs |= new_diffs
            
            if backtrack(row_idx, pos + 1, current_row):
                current_row.pop()
                used_diffs -= new_diffs
                return True
            
            current_row.pop()
            used_diffs -= new_diffs
        
        return False
    
    backtrack(0, 0, [])
    
    if best_solution is not None:
        return best_solution
    return None
# EVOLVE-BLOCK-END
