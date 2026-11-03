import time
import random
import sys

# Increase recursion limit for deep backtracking
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    # Leave 5 seconds for safety and cleanup
    deadline = start_time + 115.0
    
    # 1. Baseline: Deterministic Greedy
    # This gives us a valid solution and an upper bound on the scope.
    best_rows, best_scope = _greedy_construct(n_rows, k)
    
    if best_rows is None:
        return [] # Should not happen

    # 2. Backtracking Search to improve scope
    # We try to find a solution with a smaller scope.
    # We iterate downwards from best_scope - 1.
    
    current_best_scope = best_scope
    
    # Theoretical lower bound on scope
    # Total differences = n_rows * k * (k+1) / 2
    # These must fit in [1, M-1]
    # So M-1 >= total_diffs => M >= total_diffs + 1
    lb = (n_rows * k * (k + 1)) // 2 + 1
    
    search_scope = current_best_scope - 1
    
    # We will try to find a solution with scope <= search_scope
    # If found, we update best and try lower.
    # If not found (timeout or no solution), we decrement search_scope.
    
    while time.time() < deadline and search_scope > lb:
        # Run backtracking for a short time slice
        # We allocate a small time slice to avoid getting stuck on one scope too long
        # If the scope is impossible, we want to move on.
        # If it's possible but hard, we might need more time, but we have a global limit.
        
        # Use a time slice of up to 2 seconds, or remaining time if less.
        slice_deadline = time.time() + min(2.0, deadline - time.time())
        
        found_rows = _backtracking_solve(n_rows, k, search_scope, slice_deadline)
        
        if found_rows is not None:
            # We found a solution with scope <= search_scope
            actual_scope = max(max(row) for row in found_rows)
            if actual_scope < current_best_scope:
                current_best_scope = actual_scope
                best_rows = found_rows
                # Try to go even lower
                search_scope = actual_scope - 1
            else:
                # This case should not happen if we strictly check scope in backtracking
                # But if it does, we just continue
                pass
        else:
            # No solution found in the time slice.
            # Decrement search scope to try a tighter bound.
            # Note: If search_scope is impossible, search_scope-1 is also impossible.
            # So this effectively skips impossible scopes.
            search_scope -= 1
            
    return best_rows

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last = row[-1]
            x = last + 1
            while True:
                valid = True
                for a in row:
                    if (x - a) in used:
                        valid = False
                        break
                if valid:
                    break
                x += 1
            
            row.append(x)
            for a in row[:-1]:
                used.add(x - a)
        
        rows.append(row)
        
    return rows, max(max(row) for row in rows)

def _backtracking_solve(n_rows, k, max_scope, deadline):
    used = set()
    rows = []
    
    def backtrack(row_idx, current_row):
        nonlocal used, rows
        
        if time.time() > deadline:
            return None
        
        if row_idx == n_rows:
            # All rows constructed
            return list(rows)
        
        if len(current_row) == k + 1:
            # Row complete
            rows.append(current_row)
            result = backtrack(row_idx + 1, [0])
            if result is not None:
                return result
            rows.pop()
            return None
        
        # Try to find next element for current_row
        last = current_row[-1]
        
        candidates = []
        # We can limit the search range. 
        # The maximum possible value is max_scope.
        
        for x in range(last + 1, max_scope + 1):
            valid = True
            for a in current_row:
                diff = x - a
                if diff in used:
                    valid = False
                    break
            if valid:
                candidates.append(x)
                
        if not candidates:
            return None
            
        # Randomize candidates to escape local minima
        random.shuffle(candidates)
        
        for x in candidates:
            # Add x to current row
            current_row.append(x)
            # Update used set
            new_diffs = []
            for a in current_row[:-1]:
                diff = x - a
                new_diffs.append(diff)
                used.add(diff)
                
            # Recurse
            result = backtrack(row_idx, current_row)
            if result is not None:
                return result
            
            # Backtrack
            current_row.pop()
            for diff in new_diffs:
                used.remove(diff)
                
        return None
        
    return backtrack(0, [0])
