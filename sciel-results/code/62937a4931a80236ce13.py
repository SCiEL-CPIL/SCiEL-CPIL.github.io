import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_result = None
    best_scope = float('inf')
    
    # We will run multiple randomized searches
    # Each search tries to build the rows using backtracking
    # with a bias towards smaller marks, but allows deviations.
    
    def solve_with_backtracking(seed_val):
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []
        current_row = [0]
        
        # We will use a recursive function or a manual stack.
        # Recursive is easier to read, but Python recursion limit might be an issue if depth is high.
        # Depth is at most n_rows * k = 35. This is fine.
        
        def backtrack(row_idx, current_marks, used_set):
            nonlocal best_result, best_scope, deadline
            
            if time.time() > deadline:
                return False
            
            # If we have completed all rows
            if row_idx == n_rows:
                # Calculate scope
                scope = max(max(r) for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_result = [r[:] for r in rows]
                return True # Stop searching this branch, we found a solution
                
            # If we are in the middle of a row
            if len(current_marks) == k + 1:
                # Row complete, start next
                rows.append(current_marks[:])
                success = backtrack(row_idx + 1, [0], used_set)
                rows.pop()
                return success
            
            # Determine next mark
            last_mark = current_marks[-1]
            # We need to find x > last_mark
            # To allow backtracking, we need to iterate through candidates.
            # But we can't just iterate to infinity.
            # We can set a local limit or just try candidates in increasing order.
            # If we are backtracking, we will have already tried smaller ones.
            
            # Heuristic: Try candidates in increasing order.
            # To add randomness, we can shuffle the "order" of checking?
            # No, for minimizing scope, checking smaller x first is good.
            # Randomization can be in *which* branch to explore first if multiple valid x exist?
            # Actually, the standard backtracking will try x=last+1, then last+2, etc.
            # If we want to escape local minima, we might need to jump.
            
            # Let's try a range of candidates.
            # A good upper bound for the next mark in a row is maybe best_scope.
            # If we haven't found a solution yet, we can go up to some limit, say 200.
            max_candidate = best_scope if best_scope < float('inf') else 200
            
            found_valid = False
            # We iterate x from last_mark + 1 to max_candidate
            for x in range(last_mark + 1, max_candidate + 1):
                if time.time() > deadline:
                    return False
                
                # Check differences
                new_diffs = []
                valid = True
                for a in current_marks:
                    d = x - a
                    if d in used_set:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    # Try this x
                    current_marks.append(x)
                    used_set.update(new_diffs)
                    
                    success = backtrack(row_idx, current_marks, used_set)
                    
                    current_marks.pop()
                    for d in new_diffs:
                        used_set.remove(d)
                        
                    if success:
                        return True # Propagate success up
                    found_valid = True # We found at least one valid x, so we didn't hit a dead end immediately at this level, but deeper down might have failed.
                    # If we are not in "find one solution" mode but "find best solution", we should continue.
                    # However, the problem asks for the *best known* (minimized scope).
                    # The seed program just returns *a* valid set. The score is based on the scope.
                    # So we want to find the *minimum* scope.
                    # This implies we should NOT stop at the first solution found, unless we are doing binary search on scope.
                    # Or, we use the found solution to set a bound, and continue searching for better ones?
                    # Given the time limit, finding *a* good solution is key.
                    # If we find a solution with scope S, we can prune any branch that exceeds S.
                    # But continuing to search for a better S requires exploring other branches.
                    
                    # Let's modify the logic:
                    # 1. Find *any* solution.
                    # 2. Use its scope as a bound.
                    # 3. Continue searching with that bound to find a better one.
                    
                    # In the current structure, if `success` is True, it means we found a solution.
                    # We updated `best_scope`.
                    # We should return False here to allow the loop to continue and try other x's?
                    # No, if we return True, we stop exploring this path.
                    # If we want to find the *best* solution, we should return False to keep searching,
                    # but only if we haven't hit the time limit.
                    # However, if we find a solution, we know a valid bound.
                    # Let's return False to allow backtracking to explore other options,
                    # UNLESS we are confident this is optimal (which we aren't).
                    
                    # So, we continue the loop to try the next x.
                    # But wait, if `backtrack` returns True, it means it found a solution.
                    # We have updated `best_result`.
                    # We should NOT return True immediately to stop the whole search,
                    # because we want to minimize scope.
                    # So we continue the for loop.
            
            return False # No solution found in this branch (or time out)
        
        # Start the search
        backtrack(0, [0], set())
        
    # Run multiple seeds
    # The number of seeds depends on time.
    # Let's estimate time per seed.
    # A greedy run is very fast. Backtracking might be slower.
    # Let's do 10-20 seeds if time permits.
    
    # First, run a pure greedy to get an initial bound quickly.
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used_diffs:
                row.append(x)
                used_diffs |= new
        rows.append(row)
    greedy_scope = max(max(r) for r in rows)
    best_scope = greedy_scope
    best_result = [r[:] for r in rows]
    
    # Now run randomized backtracking searches
    # We will run them until we are close to the deadline.
    seed_val = 42
    while time.time() < deadline - 10: # Leave 10s margin
        solve_with_backtracking(seed_val)
        seed_val += 1
        # Optional: If we found a solution with scope 105 (the bound), we can stop.
        if best_scope <= 105:
            break
            
    return best_result
