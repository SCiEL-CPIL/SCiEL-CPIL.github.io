"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import sys
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry (the scope).
    
    Uses a backtracking search with dynamic variable ordering (MRV-like)
    focusing on the global uniqueness constraint on differences.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Simple counting bound: need at least total_diffs distinct differences.
    # The differences range from 1 to m where m is the largest entry.
    # So we need m >= total_diffs.
    
    start_m = total_diffs
    
    # Try increasing m until we find a valid configuration
    # We use a node limit to avoid hanging on impossible m values
    # If a valid config is found, return it.
    
    # We will try m from start_m upwards.
    # To be safe against timeouts, we limit the search effort per m.
    
    for m in range(start_m, start_m + 100): # Search up to 100 above the bound
        result = _backtracking_construct(n_rows, k, m, node_limit=500000)
        if result is not None:
            return result
            
    # Fallback: use the greedy approach if backtracking fails
    # This ensures we return *something*, though possibly suboptimal
    return _greedy_construct(n_rows, k)


def _backtracking_construct(n_rows, k, max_val, node_limit=500000):
    """
    Backtracking search to construct n_rows rows of k+1 integers in [0, max_val]
    starting at 0, with all differences distinct across all rows.
    """
    used_diffs = bytearray(max_val + 1)
    rows = []
    node_count = 0
    
    # We'll build rows one by one.
    
    def build_row(row_idx, current_row, remaining):
        """
        Build a single row starting at 0.
        current_row: list of placed values so far (starts with [0])
        remaining: number of more values to place
        Returns True if successful, False otherwise.
        """
        nonlocal node_count
        node_count += 1
        if node_count > node_limit:
            return False
        
        if remaining == 0:
            return True
        
        last_val = current_row[-1]
        
        # The next value must be > last_val and <= max_val
        # Also, we need 'remaining - 1' more values after this one,
        # so the max value for this candidate is max_val - (remaining - 1)
        max_candidate = max_val - (remaining - 1)
        
        if last_val >= max_candidate:
            return False
        
        # Heuristic: Try larger values first to spread points out
        # This often helps in Golomb ruler problems to avoid clustering
        # which consumes small differences.
        
        # Iterate from max_candidate down to last_val + 1
        for candidate in range(max_candidate, last_val, -1):
            # Check if all differences are unused
            valid = True
            new_diffs = []
            for a in current_row:
                d = candidate - a
                if used_diffs[d]:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Place the candidate
            current_row.append(candidate)
            for d in new_diffs:
                used_diffs[d] = 1
            
            if build_row(row_idx, current_row, remaining - 1):
                return True
            
            # Backtrack
            current_row.pop()
            for d in new_diffs:
                used_diffs[d] = 0
        
        return False
    
    # Build each row
    for row_idx in range(n_rows):
        current_row = [0]
        if not build_row(row_idx, current_row, k):
            return None
        rows.append(current_row)
    
    return rows


def _greedy_construct(n_rows, k):
    """Greedy fallback: same as the original seed program."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
