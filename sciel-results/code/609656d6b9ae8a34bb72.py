import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    # k=5 means 6 elements per row.
    # n_rows=7.
    # Target: Minimize max element M.
    # Lower bound: n_rows * (k+1)*k/2 = 7 * 15 = 105.
    
    if k != 5 or n_rows != 7:
        # Fallback to greedy for other parameters, though task specifies this case
        # But the interface is general. We should handle general cases if possible,
        # but the prompt says "The evaluator passes this task's n_rows and k".
        # And "Keep the function names...".
        # The seed is greedy. I will implement the specific solver for 7,5 and fallback to greedy otherwise.
        return _greedy_fallback(n_rows, k)

    M_lower = 105
    # Upper bound from literature is 112. We can search up to 120 to be safe.
    for M in range(M_lower, 121):
        # Generate all valid Golomb rulers of order 6 with max element <= M
        candidates = _generate_rulers(M, k+1)
        
        if len(candidates) < n_rows:
            continue
            
        # Try to find n_rows disjoint rulers
        # candidates is a list of (row_tuple, mask_int)
        solution = _find_disjoint_set(candidates, n_rows)
        
        if solution:
            # Convert tuples back to lists
            return [list(row) for row in solution]
            
    # If not found (should not happen), return greedy
    return _greedy_fallback(n_rows, k)

def _greedy_fallback(n_rows, k):
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

def _generate_rulers(M, order):
    """
    Generate all Golomb rulers of given order (number of marks) 
    starting with 0, with max element <= M.
    Returns list of (row_tuple, diff_mask).
    """
    rulers = []
    # order is 6 for k=5.
    # row[0] is always 0.
    
    def backtrack(idx, last_val, current_row, current_diffs_mask):
        if idx == order:
            # Check if max element is <= M (implied by loop bounds if we constrain x)
            # current_row is a tuple of length 'order'
            rulers.append((tuple(current_row), current_diffs_mask))
            return
        
        # Try next value
        # Optimization: The smallest possible next value is last_val + 1.
        # The largest possible next value is M.
        # Also, we need to ensure that the differences fit in the mask.
        # Mask bits 1..M.
        
        # We can prune if even the minimum possible differences exceed M?
        # Not really, differences are just values.
        
        for x in range(last_val + 1, M + 1):
            # Calculate new differences
            new_diffs = []
            valid = True
            for a in current_row:
                d = x - a
                if d > M: # Should not happen if x <= M and a >= 0
                    valid = False
                    break
                bit = 1 << d
                if current_diffs_mask & bit:
                    valid = False
                    break
                new_diffs.append(bit)
            
            if not valid:
                continue
                
            new_mask = current_diffs_mask
            for bit in new_diffs:
                new_mask |= bit
                
            current_row.append(x)
            backtrack(idx + 1, x, current_row, new_mask)
            current_row.pop()
            
    current_row = [0]
    backtrack(1, 0, current_row, 0)
    return rulers

def _find_disjoint_set(candidates, n_needed):
    """
    Find n_needed candidates from the list such that their masks are pairwise disjoint.
    candidates: list of (row_tuple, mask_int)
    """
    n = len(candidates)
    
    # Sort candidates? 
    # Heuristic: Candidates with "rarer" differences might be better?
    # Or just leave as is.
    
    # To speed up, we can use a recursive search with pruning.
    # State: index in candidates, current used_mask, list of chosen rows.
    
    chosen = []
    
    def search(start_idx, used_mask):
        if len(chosen) == n_needed:
            return True
        
        # Pruning: If remaining candidates cannot fill the need, return False
        # Simple bound: if (n - start_idx) < (n_needed - len(chosen)): return False
        if n - start_idx < n_needed - len(chosen):
            return False
            
        for i in range(start_idx, n):
            mask = candidates[i][1]
            if used_mask & mask:
                continue
            
            # Try this candidate
            chosen.append(candidates[i][0])
            if search(i + 1, used_mask | mask):
                return True
            chosen.pop()
            
        return False
        
    if search(0, 0):
        return chosen
    return None

