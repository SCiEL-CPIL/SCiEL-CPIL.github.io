"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set using systematic backtracking 
    with pruning to minimize the largest entry.
    """
    start_time = time.time()
    max_time = 110.0
    
    # Lower bound: n_rows * (k+1)*k/2
    lb = n_rows * (k + 1) * k // 2
    
    # Try to find a solution with max value M, starting from lb
    # We increment M until we find a solution or time runs out.
    # Given the small size, this should be fast.
    
    for M in range(lb, lb + 100): # Cap the search range reasonably
        if time.time() - start_time > max_time:
            break
            
        solution = _solve_systematic(n_rows, k, M)
        if solution is not None:
            return solution
            
    # If systematic search fails or times out, use a fallback greedy/randomized approach
    # to ensure we return *something* valid, even if not optimal.
    return _greedy_fallback(n_rows, k)

def _solve_systematic(n_rows, k, max_val):
    """
    Attempts to find a valid configuration with max value <= max_val.
    Uses systematic backtracking (DFS) trying smallest candidates first.
    Includes a node limit to avoid hanging on infeasible M.
    """
    rows = []
    used_diffs = set()
    node_limit = 20000  # Limit nodes per M to ensure we try multiple M values
    nodes_visited = 0
    
    def backtrack(row_idx, current_row, last_val):
        nonlocal nodes_visited
        
        if nodes_visited > node_limit:
            return None
            
        nodes_visited += 1
        
        # If current row is complete
        if len(current_row) == k + 1:
            if row_idx + 1 == n_rows:
                # All rows complete
                return [list(r) for r in rows + [current_row]]
            
            # Move to next row
            rows.append(list(current_row))
            result = backtrack(row_idx + 1, [0], 0)
            rows.pop()
            return result
        
        # Try to extend current_row
        # Candidates are from last_val + 1 to max_val
        # We try them in increasing order to keep values small
        for x in range(last_val + 1, max_val + 1):
            # Check if x creates duplicate differences within this row or with previous rows
            valid = True
            new_diffs = set()
            for a in current_row:
                d = x - a
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                # Tentatively add x
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                # Recurse
                result = backtrack(row_idx, current_row, x)
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
                
                if result is not None:
                    return result
        
        return None

    # Start backtracking
    return backtrack(0, [0], 0)

def _greedy_fallback(n_rows, k):
    """Simple greedy to ensure we return *something* valid."""
    rows = []
    used_diffs = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
