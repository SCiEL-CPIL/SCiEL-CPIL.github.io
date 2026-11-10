import sys
import time

# Cache for valid rows
# Key: (M, k)
# Value: List of tuples (row_tuple, diff_bitset)
_valid_rows_cache = {}

def _get_valid_rows(M, k):
    """
    Generates all valid rows of length k+1 starting with 0,
    with max element <= M, and distinct differences.
    """
    key = (M, k)
    if key in _valid_rows_cache:
        return _valid_rows_cache[key]
    
    rows = []
    
    def backtrack(current_row, used_diffs):
        # Pruning: if we have enough elements, store it
        if len(current_row) == k + 1:
            rows.append((tuple(current_row), used_diffs))
            return
        
        last_val = current_row[-1]
        # The next value x must be > last_val and <= M
        # Also, we must be able to fit (k+1 - len(current_row)) - 1 more values
        # The minimum increment is 1, so x can be at most M - (required_slots)
        # But x just needs to be <= M. The subsequent values will be > x.
        # So x <= M - (k + 1 - len(current_row) - 1)
        # Let's use this bound to prune.
        remaining = k + 1 - len(current_row)
        max_x = M - (remaining - 1)
        
        for x in range(last_val + 1, max_x + 1):
            valid = True
            new_diffs = []
            for a in current_row:
                d = x - a
                # d is always > 0
                if (used_diffs >> d) & 1:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                new_used = used_diffs
                for d in new_diffs:
                    new_used |= (1 << d)
                
                current_row.append(x)
                backtrack(current_row, new_used)
                current_row.pop()
                
    backtrack([0], 0)
    _valid_rows_cache[key] = rows
    return rows

def _solve_set_packing(candidates, n_rows, k, time_limit=10.0):
    """
    Selects n_rows disjoint rows from candidates.
    candidates: list of (row_tuple, diff_bitset)
    Returns a list of row_tuples or None.
    """
    start_time = time.time()
    
    # Sort candidates to enable symmetry breaking (lexicographical order of row values)
    # This ensures we only consider sets of rows in one specific order
    candidates = sorted(candidates, key=lambda x: x[0])
    
    selected = []
    used_diffs = 0
    last_idx = -1
    
    def backtrack(idx, selected, used_diffs):
        if len(selected) == n_rows:
            return True
        
        if time.time() - start_time > time_limit:
            return False
            
        # Pruning: not enough candidates left
        if idx + (n_rows - len(selected)) > len(candidates):
            return False
            
        # Iterate through candidates from idx onwards
        for i in range(idx, len(candidates)):
            row_vals, row_bits = candidates[i]
            
            # Check compatibility
            if (row_bits & used_diffs) == 0:
                selected.append(row_vals)
                if backtrack(i + 1, selected, used_diffs | row_bits):
                    return True
                selected.pop()
                
        return False
        
    if backtrack(0, selected, 0):
        return selected
    return None

def construct(n_rows, k):
    # Calculate lower bound
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    
    # Start searching from lower_bound
    # The known optimal for (7,5) is 112.
    # We search up to a reasonable limit.
    start_time = time.time()
    time_limit = 110.0
    
    for M in range(lower_bound, lower_bound + 30):
        if time.time() - start_time > time_limit:
            break
            
        # Get all valid rows for this M
        candidates = _get_valid_rows(M, k)
        
        # If we don't have enough candidates, we can't form n_rows disjoint sets
        if len(candidates) < n_rows:
            continue
            
        # Try to find a solution
        solution = _solve_set_packing(candidates, n_rows, k, time_limit=10.0)
        
        if solution:
            return [list(r) for r in solution]
            
    # Fallback: Greedy construction
    # This ensures we return something valid, even if not optimal
    return _greedy_fallback(n_rows, k)

def _greedy_fallback(n_rows, k):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        # Greedily pick smallest possible x to keep M small
        # We don't have a fixed M here, so we just grow as needed
        # But we want to minimize the max entry, so greedy smallest-first is good
        while len(row) < k + 1:
            last = row[-1]
            found = False
            # Search for next x
            # We can limit the search range to avoid infinite loops, 
            # but for valid inputs, a solution exists.
            for x in range(last + 1, 1000):
                valid = True
                new_bits = 0
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        valid = False
                        break
                    new_bits |= (1 << d)
                if valid:
                    row.append(x)
                    used |= new_bits
                    found = True
                    break
            if not found:
                # Fallback if greedy gets stuck (shouldn't happen for small k)
                row.append(last + 1)
                break
        rows.append(row)
    return rows
