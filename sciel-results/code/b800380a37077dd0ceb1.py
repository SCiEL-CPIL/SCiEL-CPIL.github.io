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
    # So we need m >= total_diffs. But we can do better with tighter bounds.
    
    # Start with a reasonable lower bound and search upward
    # The simple counting bound is m >= total_diffs
    # But we also need each row to have k+1 elements in [0, m], so C(m+1, 2) >= total_diffs per row isn't quite right.
    # Actually, for a single row of k+1 elements in [0, m], the max number of distinct differences is k*(k+1)/2.
    # The minimum possible maximum for a single row (a Golomb ruler of order k+1) is known to be around k^2.
    
    # Let's use a heuristic lower bound
    lower_bound = total_diffs
    # Also, each row needs at least a Golomb ruler of length k+1
    # The shortest Golomb ruler of order n has length at least n(n-1)/2 for small n, but asymptotically ~ n^2
    # For k+1 marks, the minimum length is at least k*(k+1)/2... no, that's the number of differences.
    # The minimum span is at least k for k+1 points, but much more for Golomb rulers.
    # Let's just start from a reasonable estimate
    
    # Use a simple estimate: for a Golomb ruler of order k+1, the length is at least ~k^2/2
    # But we have n_rows such rulers sharing the difference space.
    # Let's start from max(total_diffs, k*(k+1)//2 + 10) and go up
    
    start_m = max(total_diffs, k * (k + 1) // 2 + 5)
    
    # Try increasing m until we find a valid configuration
    m = start_m
    max_attempts = 100  # Don't try too many m values
    
    for attempt in range(max_attempts):
        current_m = start_m + attempt
        # Try to find a valid configuration with maximum entry <= current_m
        result = _backtracking_construct(n_rows, k, current_m)
        if result is not None:
            return result
        # If we've been trying for a while, increase m more aggressively
        if attempt > 20:
            m = current_m + 5
        else:
            m = current_m + 1
    
    # Fallback: use the greedy approach if backtracking fails
    return _greedy_construct(n_rows, k)


def _backtracking_construct(n_rows, k, max_val):
    """
    Backtracking search to construct n_rows rows of k+1 integers in [0, max_val]
    starting at 0, with all differences distinct across all rows.
    """
    used_diffs = set()
    rows = []
    
    # We'll build rows one by one. For each row, we need to place k+1 points
    # starting at 0, with all differences unique globally.
    
    # For efficiency, we can precompute valid placements for each row
    # using a recursive search with pruning
    
    def build_row(row_idx, current_row, pos, remaining):
        """
        Build a single row starting at 0.
        current_row: list of placed values so far (starts with [0])
        pos: current position in the row
        remaining: number of more values to place
        """
        if remaining == 0:
            return True
        
        # Try to find the next value
        # MRV-like: try values that have the fewest conflicts
        # For simplicity, try values in increasing order but with pruning
        
        last_val = current_row[-1]
        
        # The next value must be > last_val and <= max_val
        # Also, all differences with existing values must not be in used_diffs
        
        for candidate in range(last_val + 1, max_val + 1):
            # Check if all differences are unused
            valid = True
            new_diffs = []
            for a in current_row:
                d = candidate - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Also check that the new differences don't conflict with each other
            # (they can't since they're all different from candidate - a for different a)
            
            # Pruning: check if we can still complete the row
            # The remaining values must be placed in (candidate, max_val]
            # and we need 'remaining - 1' more values after this one
            if max_val - candidate < remaining - 1:
                break  # Not enough room
            
            # Place the candidate
            current_row.append(candidate)
            used_diffs.update(new_diffs)
            
            if build_row(row_idx, current_row, pos + 1, remaining - 1):
                return True
            
            # Backtrack
            current_row.pop()
            used_diffs.difference_update(new_diffs)
        
        return False
    
    # Build each row
    for row_idx in range(n_rows):
        current_row = [0]
        if not build_row(row_idx, current_row, 1, k):
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
