import time
import sys
import random
from functools import lru_cache

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows and k+1 entries per row.
    Returns a list of rows.
    """
    deadline = time.time() + 140.0  # Safety margin
    best_scope = float('inf')
    best_rows = None
    
    # Total number of differences required
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    
    # Initialize rows with 0
    rows = [[0] + [None] * k for _ in range(n_rows)]
    
    # List of all variables to be assigned: (row_idx, col_idx)
    # We exclude col 0 as it is fixed to 0
    variables = [(r, c) for r in range(n_rows) for c in range(1, k + 1)]
    
    # Precompute all possible differences for a row
    # diff(r, i, j) = x[r][j] - x[r][i]
    
    def is_valid_assignment(rows, r, c, val):
        """
        Check if assigning val to rows[r][c] is valid.
        1. Strictly increasing: rows[r][c-1] < val < rows[r][c+1] (if exists)
        2. No duplicate differences: for all i < c, val - rows[r][i] must not be in used_diffs.
           Also, for all j > c, rows[r][j] - val must not be in used_diffs (if rows[r][j] is assigned).
        """
        # Check strict increase
        if c > 0 and val <= rows[r][c-1]:
            return False
        if c < k and rows[r][c+1] is not None and val >= rows[r][c+1]:
            return False
            
        # Check differences
        # Differences with previous assigned entries in the same row
        for i in range(c):
            d = val - rows[r][i]
            if d in used_diffs:
                return False
            # Note: We don't add to used_diffs here, just check
        
        # Differences with next assigned entries in the same row
        for j in range(c + 1, k + 1):
            if rows[r][j] is not None:
                d = rows[r][j] - val
                if d in used_diffs:
                    return False
        
        return True

    used_diffs = set()
    
    def backtrack(var_idx, current_scope):
        nonlocal best_scope, best_rows, used_diffs
        
        if time.time() > deadline:
            return False
            
        if var_idx == len(variables):
            # All variables assigned
            # Calculate actual scope
            actual_scope = max(max(row) for row in rows)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = [row[:] for row in rows]
                # We can continue searching for a better scope if we have time
                # But if we found the theoretical lower bound, we could stop.
                # Lower bound is roughly sqrt(total_diffs) or similar, but 134 is known.
            return True
            
        r, c = variables[var_idx]
        
        # Determine possible values for rows[r][c]
        # Lower bound: rows[r][c-1] + 1
        lower = rows[r][c-1] + 1 if c > 0 else 1
        # Upper bound: 
        # 1. If next is assigned: rows[r][c+1] - 1
        # 2. Else: current best scope (or some heuristic limit)
        if c < k and rows[r][c+1] is not None:
            upper = rows[r][c+1] - 1
        else:
            # Heuristic upper bound to prune search
            # If we are at the last column of the last row, we can go up to best_scope - 1
            # Otherwise, we should be careful.
            # A safe upper bound is best_scope - 1, but we might need to go higher if best_scope is inf.
            # Let's use a dynamic bound.
            upper = best_scope - 1 if best_scope < float('inf') else 500
            
        if lower > upper:
            return False
            
        # Generate candidates
        # To improve MRV, we could check how many candidates are valid, but that's expensive.
        # Instead, we iterate and check.
        # Randomize order to avoid deterministic traps? 
        # For backtracking, deterministic is often fine if heuristics are good.
        # Let's try increasing order first, as smaller values are generally better for scope.
        
        candidates = list(range(lower, upper + 1))
        
        # Optional: Shuffle if we want diversity, but for finding the *smallest* scope, 
        # trying small values first is a good strategy (branch and bound).
        
        for val in candidates:
            if not is_valid_assignment(rows, r, c, val):
                continue
                
            # Tentatively assign
            rows[r][c] = val
            
            # Update used_diffs
            # Add differences with previous
            new_diffs = []
            for i in range(c):
                d = val - rows[r][i]
                new_diffs.append(d)
                used_diffs.add(d)
                
            # Add differences with next (already assigned)
            for j in range(c + 1, k + 1):
                if rows[r][j] is not None:
                    d = rows[r][j] - val
                    new_diffs.append(d)
                    used_diffs.add(d)
                    
            # Recurse
            if backtrack(var_idx + 1, val):
                # Check if we found a solution better than current best
                # Actually, the recursion updates best_scope.
                # If we found a solution with scope <= current upper bound, we can prune?
                # No, we want the *minimum*.
                pass
            else:
                # Backtrack
                pass
                
            # Undo assignment
            rows[r][c] = None
            for d in new_diffs:
                used_diffs.discard(d)
                
            # If we found a solution with scope equal to the theoretical lower bound, we could stop.
            # But we don't know the exact lower bound easily.
            
        return False

    # Variable ordering: 
    # Standard order (row by row) might be suboptimal.
    # MRV (Minimum Remaining Values) is dynamic.
    # For a static implementation, let's stick to row-by-row, column-by-column.
    # However, we can improve by choosing the variable with the tightest constraints first.
    # Since we can't easily compute MRV dynamically in a simple recursive call without overhead,
    # let's use a heuristic: assign variables in a way that connects rows?
    # Actually, for small n, row-by-row is often sufficient.
    
    # Let's try a slightly different variable ordering: interleave columns?
    # No, strict increase within a row is a strong local constraint.
    
    # To speed up, we can use a simpler greedy initialization to get a good initial best_scope,
    # then use backtracking to improve it.
    
    # Greedy Initialization
    def greedy_init():
        nonlocal best_scope, best_rows, used_diffs
        rows_g = [[0] + [0] * k for _ in range(n_rows)]
        used_g = set()
        
        for r in range(n_rows):
            for c in range(1, k + 1):
                found = False
                # Try small values first
                for val in range(rows_g[r][c-1] + 1, 500):
                    valid = True
                    for i in range(c):
                        d = val - rows_g[r][i]
                        if d in used_g:
                            valid = False
                            break
                    if valid:
                        rows_g[r][c] = val
                        for i in range(c):
                            used_g.add(val - rows_g[r][i])
                        found = True
                        break
                if not found:
                    return None, float('inf')
                    
        scope_g = max(max(row) for row in rows_g)
        if scope_g < best_scope:
            best_scope = scope_g
            best_rows = [row[:] for row in rows_g]
        return rows_g, scope_g

    # Run greedy to get initial bound
    _, init_scope = greedy_init()
    
    # Now run backtracking to find a better solution
    # We need to reset the state for backtracking
    rows = [[0] + [None] * k for _ in range(n_rows)]
    used_diffs = set()
    
    # We want to find a solution with scope < init_scope
    # So we can limit the search to values < init_scope
    
    # To make backtracking more effective, we can use a different variable order.
    # Let's try to assign the last column of the first row first? No, that's hard.
    # Let's stick to standard order but with the upper bound set to init_scope - 1.
    
    # We'll use a global flag to stop early if we find a very good solution
    found_better = [False]
    
    def backtrack_optimized(var_idx):
        nonlocal best_scope, best_rows, used_diffs, found_better
        
        if time.time() > deadline:
            return False
            
        if found_better[0]:
            return False
            
        if var_idx == len(variables):
            actual_scope = max(max(row) for row in rows)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = [row[:] for row in rows]
                # If we found the known best (134), we can stop?
                # Or if we found something close to the lower bound.
                if actual_scope <= 134:
                    found_better[0] = True
            return True
            
        r, c = variables[var_idx]
        
        lower = rows[r][c-1] + 1 if c > 0 else 1
        if c < k and rows[r][c+1] is not None:
            upper = rows[r][c+1] - 1
        else:
            upper = best_scope - 1
            
        if lower > upper:
            return False
            
        # Heuristic: Try values that are likely to be valid
        # We can check validity quickly
        
        for val in range(lower, upper + 1):
            if not is_valid_assignment(rows, r, c, val):
                continue
                
            rows[r][c] = val
            
            new_diffs = []
            for i in range(c):
                d = val - rows[r][i]
                new_diffs.append(d)
                used_diffs.add(d)
                
            for j in range(c + 1, k + 1):
                if rows[r][j] is not None:
                    d = rows[r][j] - val
                    new_diffs.append(d)
                    used_diffs.add(d)
                    
            if backtrack_optimized(var_idx + 1):
                if found_better[0]:
                    # Cleanup before returning
                    rows[r][c] = None
                    for d in new_diffs:
                        used_diffs.discard(d)
                    return True
            else:
                pass
                
            rows[r][c] = None
            for d in new_diffs:
                used_diffs.discard(d)
                
        return False

    # Start backtracking
    backtrack_optimized(0)
    
    if best_rows is None:
        # Fallback to greedy if backtracking failed or timed out without finding anything
        rows_g, scope_g = greedy_init()
        if rows_g is not None:
            best_rows = rows_g
            best_scope = scope_g
        else:
            # Last resort: pure greedy with wider range
            rows_g = [[0] for _ in range(n_rows)]
            used_g = set()
            for r in range(n_rows):
                for c in range(1, k + 1):
                    for val in range(rows_g[r][c-1] + 1, 1000):
                        valid = True
                        for i in range(c):
                            d = val - rows_g[r][i]
                            if d in used_g:
                                valid = False
                                break
                        if valid:
                            rows_g[r].append(val)
                            for i in range(c):
                                used_g.add(val - rows_g[r][i])
                            break
            best_rows = rows_g

    return best_rows

# Helper to run the construct function for testing
if __name__ == "__main__":
    # Test with the specific case
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
    print("Rows:", rows)
    
    # Verify validity
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    print("Invalid: duplicate diff", d)
                diffs.add(d)
    print("Total diffs:", len(diffs))
    print("Expected diffs:", 4 * (8 * 7 // 2))
