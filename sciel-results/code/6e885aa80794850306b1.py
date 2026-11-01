import time
import sys

def construct(n_rows, k):
    """
    Finds a difference triangle set of n_rows rows, each with k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 100.0  # Safety margin for 120s limit
    
    # Lower bound on the max value M
    # Total distinct differences needed = n_rows * (k*(k+1)/2)
    # These differences must be in {1, ..., M}
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    lower_bound = total_diffs_needed
    
    # We will try to find a solution with max value M
    # Starting from lower_bound
    best_solution = None
    best_M = float('inf')
    
    # Pre-compute combinations of indices for a row of size k+1
    # The row is [0, x1, x2, ..., xk]
    # We need to choose x1...xk from 1..M such that 0 < x1 < ... < xk <= M
    # And the differences are unique and not in used_diffs
    
    def backtrack(row_idx, current_rows, used_diffs, M):
        nonlocal best_solution, best_M
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            # Found a solution
            best_solution = [list(r) for r in current_rows]
            best_M = M
            return True
        
        # Pruning: Check if enough differences are left
        # Diffs needed for remaining rows
        remaining_rows = n_rows - row_idx
        diffs_needed = remaining_rows * (k * (k + 1) // 2)
        # Diffs available in 1..M
        # used_diffs contains all diffs used so far.
        # We need to count how many numbers in 1..M are NOT in used_diffs.
        # Since M can be up to ~150, and used_diffs size is small, 
        # we can just iterate or use a set difference.
        # For efficiency, we can maintain a bitmask or just check count.
        # For small M, iterating 1..M is fine.
        available = 0
        # Optimization: if M is small, just count.
        # used_diffs is a set.
        for d in range(1, M + 1):
            if d not in used_diffs:
                available += 1
                if available >= diffs_needed:
                    break
        
        if available < diffs_needed:
            return False
        
        # Try to build the next row
        # The row starts with 0.
        # We need to choose k numbers from 1..M.
        # To reduce symmetry, we can assume the rows are sorted by their last element?
        # Or just try all combinations.
        # Number of combinations C(M, k) can be large if M is large.
        # But M starts near lower bound.
        
        # We iterate through possible last elements of the row?
        # Or generate combinations of k elements from 1..M.
        
        # A smarter way:
        # We know the row must be increasing: 0 = a0 < a1 < ... < ak <= M
        # Let's pick a1...ak.
        
        # To speed up, we can fix the order of rows by their max value.
        # i.e., max(row[i]) <= max(row[i+1])
        # This helps prune.
        
        last_max = current_rows[-1][-1] if current_rows else 0
        
        # Generate candidate rows
        # We can use a recursive helper to build the row element by element
        # to allow pruning within the row construction.
        
        def build_row(pos, current_row, row_diffs, max_allowed):
            nonlocal used_diffs
            if time.time() - start_time > time_limit:
                return None
            
            if pos == k:
                # Row complete
                # Check if valid (diffs already checked during build)
                # And check max constraint
                if current_row[-1] <= M and current_row[-1] >= last_max:
                    # Try to place this row
                    new_used = used_diffs | row_diffs
                    current_rows.append(current_row)
                    if backtrack(row_idx + 1, current_rows, new_used, M):
                        return current_row
                    current_rows.pop()
                return None
            
            # Determine start value for next element
            # If pos == 0, start from 1
            # Else start from current_row[-1] + 1
            start_val = 1 if pos == 0 else current_row[-1] + 1
            
            # Max value for this element
            # It must be <= M
            # Also, the remaining k-pos elements must fit in M
            # So current_val + (k - pos) <= M  => current_val <= M - (k - pos)
            max_val = M - (k - pos)
            
            for val in range(start_val, max_val + 1):
                # Calculate new diffs
                new_diffs = set()
                valid = True
                for prev in current_row:
                    diff = val - prev
                    if diff in used_diffs or diff in row_diffs or diff in new_diffs:
                        valid = False
                        break
                    new_diffs.add(diff)
                
                if not valid:
                    continue
                
                # Prune within row: 
                # The diffs generated so far plus future diffs must be distinct.
                # This is handled by checking against used_diffs and row_diffs.
                
                result = build_row(pos + 1, current_row + [val], row_diffs | new_diffs, max_allowed)
                if result is not None:
                    return result
            
            return None
        
        # Start building the row
        # We need to ensure the row's max is >= last_max to maintain sorted order
        # This is checked in the base case of build_row.
        
        res = build_row(0, [0], set(), M)
        if res is not None:
            return True
        
        return False

    # Search for the smallest M
    # We start from lower_bound
    # We might need to go up.
    
    # To handle the case where the search is slow, we can run it in a loop with increasing M.
    # But backtracking for a fixed M is a decision problem.
    
    # Let's try M from lower_bound up to some limit (e.g., lower_bound + 50)
    # If we find a solution, we return it.
    # If time runs out, we return the best found (which might be None if no solution found yet, 
    # in which case we should return a valid greedy solution).
    
    # Fallback greedy solution
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    fallback = greedy_construct()
    
    # Try to improve
    # We will iterate M
    # Note: The backtracking might be slow. We add a time check in the loop.
    
    # Let's try a few M values
    # If n_rows and k are small, lower_bound is tight.
    
    for M in range(lower_bound, lower_bound + 100):
        if time.time() - start_time > time_limit:
            break
            
        # Reset state for each M
        # We use a fresh used_diffs set
        # We also need to reset best_solution for this M search?
        # No, we want the global best.
        
        # To avoid re-using the nonlocal variables incorrectly in the nested function scope
        # we pass M as argument.
        
        # We need to clear the current_rows and used_diffs for each M attempt.
        # The backtrack function uses nonlocal best_solution.
        
        # We should probably wrap the search in a way that it stops if found.
        
        # The current implementation of backtrack returns True if found.
        # We call it with empty rows and empty used_diffs.
        
        # However, the `build_row` and `backtrack` are defined inside `construct`, 
        # so they capture `n_rows`, `k`, `start_time`, `time_limit`, `best_solution`, `best_M`.
        
        # We need to be careful: `backtrack` modifies `current_rows` and `used_diffs` (passed by ref? No, sets are mutable, but we pass copies or modify in place?).
        # In `backtrack`, we do `new_used = used_diffs | row_diffs`. This creates a new set.
        # `current_rows.append(current_row)`. This modifies the list.
        # So we must pass a fresh list and set for each top-level call.
        
        initial_rows = []
        initial_used = set()
        
        found = backtrack(0, initial_rows, initial_used, M)
        
        if found:
            # We found a solution with this M
            # Since we iterate M upwards, this is the best M found so far.
            # We can stop searching for larger M? 
            # Yes, because we want to minimize M.
            # But wait, is the first solution found for a given M necessarily the best structure?
            # Yes, M is the objective. Smaller M is better.
            # So if we find a valid set for M, we can return it immediately?
            # Yes, because we are iterating M from low to high.
            # The first valid M is the optimal M (or close to it if we skipped some? No, we iterate all).
            # So we can return best_solution.
            
            # One caveat: The search might be incomplete for a given M due to time limit?
            # If `backtrack` returns False due to time limit, `found` is False.
            # If it returns True, it found a complete solution.
            
            return best_solution

    # If no solution found with the improved search within time limit,
    # return the greedy solution.
    return fallback
