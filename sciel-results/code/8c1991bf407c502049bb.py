import time
import sys

# EVOLVE-BLOCK-START

def _greedy_construct(n_rows, k):
    """Fast greedy construction to get an initial upper bound."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Check if x is valid
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in used:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    return rows

def _backtrack_search(n_rows, k, max_scope, used_diffs, current_rows, current_row, row_idx, deadline):
    """
    Recursive backtracking to find a valid set of rows 
    with all entries <= max_scope.
    Returns True if a solution is found, updating current_rows.
    """
    if time.time() > deadline:
        return False
    
    if row_idx == n_rows:
        # All rows constructed
        return True

    if len(current_row) == k + 1:
        # Current row is complete
        # Check if we have exceeded max_scope (should be checked during construction)
        # Move to next row
        return _backtrack_search(n_rows, k, max_scope, used_diffs, 
                                 current_rows + [current_row], 
                                 [0], row_idx + 1, deadline)

    # Need to pick next element for current_row
    last_val = current_row[-1]
    # Start from last_val + 1
    # Optimization: We can bound the start value.
    # The remaining marks must fit in max_scope.
    
    # Heuristic: Try values in increasing order.
    # To break symmetry between rows, we could enforce some ordering, 
    # but since rows are indistinguishable, we just need *a* solution.
    
    for x in range(last_val + 1, max_scope + 1):
        # Check differences
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
            
        # Additional pruning: 
        # The maximum possible difference for the rest of the row 
        # must be achievable.
        # Simple bound: if x is too small, we might not be able to 
        # fit the remaining k - len(current_row) + 1 marks?
        # Actually, since we just need *distinct* differences, 
        # and we have space up to max_scope, 
        # the main constraint is the used_diffs set.
        
        # Recurse
        # Copy used_diffs to avoid mutation (or use set difference on backtrack)
        # For efficiency, we can pass a new set. 
        # k is small (5), n_rows is small (7). 
        # Total diffs per row: 15. Total diffs: 105.
        # Set size is small.
        
        new_used = used_diffs.union(new_diffs)
        
        if _backtrack_search(n_rows, k, max_scope, new_used, 
                             current_rows, 
                             current_row + [x], 
                             row_idx, deadline):
            return True
            
    return False

def construct(n_rows, k):
    # Get initial solution using greedy
    initial_rows = _greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in initial_rows)
    best_rows = initial_rows
    
    start_time = time.time()
    # Leave 10s buffer for the 120s limit
    deadline = start_time + 110
    
    # Try to improve the scope
    # We search for solutions with scope < best_scope
    target_scope = best_scope - 1
    
    while target_scope > 0:
        if time.time() > deadline:
            break
            
        # Reset search state
        # We search for a solution where all entries <= target_scope
        # used_diffs starts empty
        # current_rows starts empty
        # current_row starts with [0]
        
        # Optimization: 
        # We can order the rows. Since rows are symmetric, 
        # we can enforce that the first non-zero element of row i 
        # is >= the first non-zero element of row i-1? 
        # Or simply that the set of differences is lexicographically larger?
        # For now, simple backtracking.
        
        # To make it faster, we can try to fill rows in a specific order.
        # But standard backtracking on the first row is fine.
        
        # We pass an empty list for current_rows initially
        found = _backtrack_search(n_rows, k, target_scope, set(), [], [0], 0, deadline)
        
        if found:
            # We found a solution with this target_scope
            # We need to extract it from the recursion? 
            # The function above returns True/False but doesn't return the rows directly 
            # in a clean way for the caller to update best_rows if we don't modify in place 
            # or return the rows.
            
            # Let's modify the search to return the rows if found.
            # Actually, the current implementation updates `current_rows` in the success case 
            # but doesn't return it. 
            # Let's rewrite the search to return the rows.
            
            # Re-implementing search to return rows
            pass 
            
        # Since the previous implementation was a bit messy with state, 
        # let's implement a cleaner version that returns the solution.
        
        # We will break out of this loop and use a cleaner implementation below.
        break

    # The above while loop structure is okay, but the search function needs to return the solution.
    # Let's replace the search logic with a version that returns the list of rows.
    
    # Re-initialize for the improved search
    best_scope = max(max(row) for row in initial_rows)
    best_rows = initial_rows
    
    deadline = start_time + 110
    
    target_scope = best_scope - 1
    
    while target_scope > 0 and time.time() < deadline:
        
        def search(rows_so_far, used_diffs, current_row, row_idx):
            nonlocal deadline
            
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                # All rows done
                return rows_so_far
            
            if len(current_row) == k + 1:
                # Row complete
                # Move to next row
                return search(rows_so_far + [current_row], used_diffs, [0], row_idx + 1)
            
            last_val = current_row[-1]
            
            # Pruning: 
            # If we have remaining marks to place, 
            # and the current last_val is already high, 
            # we might not fit? 
            # But we have space up to target_scope.
            
            for x in range(last_val + 1, target_scope + 1):
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
                
                new_used = used_diffs | set(new_diffs)
                
                result = search(rows_so_far, new_used, current_row + [x], row_idx)
                if result is not None:
                    return result
            
            return None
        
        # Start search
        # We can add symmetry breaking: 
        # The first row's first non-zero element (a_1) can be assumed to be 1? 
        # No, not necessarily. But we can assume the set of differences 
        # for row 0 is the "lexicographically smallest" among all rows?
        # That's hard to enforce.
        
        # Simple search:
        solution = search([], set(), [0], 0)
        
        if solution is not None:
            # Found a solution with scope <= target_scope
            # The actual scope is max(max(row) for row in solution)
            actual_scope = max(max(row) for row in solution)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = solution
            # Try to find even smaller
            target_scope = actual_scope - 1
        else:
            # No solution with this target_scope
            target_scope -= 1
            # If we fail to find a solution for target_scope, 
            # we might want to try target_scope - 1? 
            # Actually, if no solution exists with max entry <= T, 
            # then no solution exists with max entry <= T-1.
            # So we can stop? 
            # Wait, the greedy solution gave us a valid one with best_scope.
            # If we fail at best_scope - 1, it means best_scope is the minimum?
            # Not necessarily, because we might have missed a solution due to search order?
            # No, backtracking is exhaustive. If it returns None, no solution exists 
            # with max entry <= target_scope.
            # So we can break.
            break

    return best_rows

# EVOLVE-BLOCK-END
