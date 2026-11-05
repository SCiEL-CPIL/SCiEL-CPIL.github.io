import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # 1. Greedy Initialization
    # The seed program's greedy is okay for a start, but we can randomize the order of rows or the choice of next element to get diverse starting points.
    # Actually, for (7,5) or similar, the greedy might be quite good.
    
    used_diffs = set()
    rows = []
    
    # Generate initial rows
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Try to find the smallest next element
            # We can bound the search or just go up
            # To speed up, we can estimate a bound
            while True:
                current_max += 1
                new_diffs = {current_max - x for x in row}
                if not (new_diffs & used_diffs):
                    # Check internal uniqueness
                    # Since row is increasing, and we add one element, 
                    # we just need to ensure new_diffs don't intersect with existing internal diffs?
                    # No, the 'used_diffs' set in the seed program accumulates ALL diffs from ALL rows.
                    # So if new_diffs is disjoint from used_diffs, it is disjoint from internal diffs of this row (which are in used_diffs) 
                    # AND disjoint from diffs of other rows.
                    # Wait, the seed program adds diffs to 'used' immediately.
                    # So yes, 'new_diffs & used_diffs' == empty ensures global uniqueness.
                    row.append(current_max)
                    used_diffs |= new_diffs
                    found = True
                    break
            if not found:
                # This shouldn't happen if we go to infinity, but for safety
                break
        rows.append(row)
        
    best_rows = [r[:] for r in rows]
    best_score = max(max(r) for r in best_rows)
    
    # 2. Iterative Local Search
    # We try to improve rows one by one.
    # For a given row r, we want to find a new row r' such that:
    # 1. r' starts with 0, has k+1 elements, strictly increasing.
    # 2. Diffs of r' are disjoint from Diffs of (all rows except r).
    # 3. Diffs of r' are internally unique.
    # 4. max(r') < max(r) (or we just find the minimal possible max(r'))
    
    def get_used_diffs_except_row(rows, idx):
        diffs = set()
        for i, r in enumerate(rows):
            if i == idx:
                continue
            for j in range(len(r)):
                for l in range(j):
                    diffs.add(r[j] - r[l])
        return diffs
    
    def solve_row(k, forbidden_diffs, max_bound=None):
        """
        Finds a row of length k+1 starting at 0.
        Diffs must be disjoint from forbidden_diffs.
        Diffs must be internally unique.
        Returns the row with minimal last element, or None if not found within max_bound.
        If max_bound is None, it finds the minimal possible (greedy-like but backtracking).
        """
        # We want to minimize the last element.
        # We can do a BFS/DFS searching for the smallest possible last element.
        # Since k is small, we can just try to build the row with the smallest possible values.
        # This is essentially the greedy algorithm for a single row with a fixed set of forbidden diffs.
        # Is greedy optimal for a single row given forbidden diffs?
        # Yes, for Golomb rulers (single row), greedy is NOT always optimal for minimal length, but often close.
        # However, backtracking is safer.
        
        # Let's use a recursive backtracking that tries to minimize the next element.
        # To ensure we find the *minimal* last element, we should search for a solution with last element < current_best.
        # But here we don't have a current_best for the row, we just want the best possible given the constraints.
        # We can use a branch and bound where we keep track of the current path and prune if it's already worse than a known solution?
        # Since we don't have a known solution, we just find the first valid one?
        # No, "first valid" in a DFS that tries smallest values first IS the greedy solution.
        # Greedy might not be optimal.
        
        # Let's implement a simple backtracking that tries values in increasing order.
        # To avoid infinite loops, we need a bound.
        # A safe bound is the current row's max element.
        
        best_row = None
        best_last = float('inf')
        
        def backtrack(row, used_local_diffs):
            nonlocal best_row, best_last
            
            if len(row) == k + 1:
                if row[-1] < best_last:
                    best_row = row[:]
                    best_last = row[-1]
                return True # Found a better one, but we want to find even better? 
                # Actually, if we search in increasing order of values, the first solution found 
                # for a specific depth might not be the global minimum if we don't explore all branches.
                # But if we prune branches that exceed best_last, we can find the minimum.
            
            current_max = row[-1]
            # We can start searching from current_max + 1
            # We can stop if current_max + 1 >= best_last (since we need at least 1 step up)
            # Actually, we need k - len(row) + 1 more steps.
            # So min next is current_max + 1.
            # Max next is best_last - (k - len(row)) to leave room for remaining elements?
            # Not exactly, but a good heuristic.
            
            start_val = current_max + 1
            # Heuristic upper bound for this step to stay under best_last
            # If we pick x, the remaining elements must be > x.
            # So x + (k - len(row)) < best_last is a necessary condition for a better solution.
            end_val = best_last - (k - len(row))
            
            for x in range(start_val, end_val + 1):
                new_diffs = {x - a for a in row}
                # Check against forbidden
                if new_diffs & forbidden_diffs:
                    continue
                # Check against local
                if new_diffs & used_local_diffs:
                    continue
                
                row.append(x)
                used_local_diffs |= new_diffs
                
                if backtrack(row, used_local_diffs):
                    # If we found a solution, we can return immediately if we are just looking for *any* improvement?
                    # No, we want the *best* improvement.
                    # However, returning early might miss a better solution in a different branch.
                    # But since we iterate x in increasing order, and we prune based on best_last,
                    # the first solution found at a deeper level with a smaller last element is good.
                    # To be thorough, we should continue searching to see if we can get an even smaller last element.
                    # But this might be slow.
                    # Let's return True to indicate we found a better solution, 
                    # and let the caller decide if it wants to continue.
                    # Actually, standard backtracking for optimization usually continues to find the global min.
                    # But given time constraints, finding *a* good solution is better than none.
                    # Let's just return True to stop this branch, but we don't stop the whole search 
                    # unless we want the very first improvement.
                    # Let's change strategy: Find the minimal possible last element.
                    # We can binary search on the last element? No.
                    # Let's just return True and rely on the fact that we try smaller x first.
                    # If we find a solution with last element L, we update best_last.
                    # Then we continue the loop for x? No, if we return True, we exit the function.
                    # We should NOT return True immediately if we want the global min for this row.
                    # We should return False to continue searching for better solutions?
                    # But we need to know if we found *any* solution to update best_row.
                    
                    # Let's restructure:
                    # We don't return a boolean "found better". We just update nonlocal variables.
                    # We return True if we found a solution (to potentially prune? No).
                    # Let's just let it run.
                    pass
                
                row.pop()
                # used_local_diffs is a set, we need to remove the diffs we added.
                # Since we are in a loop, it's easier to pass a copy or use a stack.
                # Given k is small, copying the set is fine.
                # Let's rewrite backtrack to use copies for safety and simplicity.
                
            return False # No solution found in this branch
        
        # Rewrite backtrack to be cleaner
        def solve():
            nonlocal best_row, best_last
            best_row = None
            best_last = float('inf')
            
            def dfs(row, local_diffs):
                nonlocal best_row, best_last
                if len(row) == k + 1:
                    if row[-1] < best_last:
                        best_row = row[:]
                        best_last = row[-1]
                    return
                
                current_max = row[-1]
                # Pruning: if current_max + (k + 1 - len(row)) >= best_last, we can't beat best_last
                if current_max + (k + 1 - len(row)) >= best_last:
                    return
                
                # Try next values
                # We can limit the range. 
                # A reasonable upper bound for the next element is best_last - (k - len(row))
                # But we also have forbidden diffs.
                
                # To speed up, we can estimate the minimum possible value.
                # Just iterate from current_max + 1 upwards.
                # We stop when x > best_last - (k - len(row))
                
                limit = best_last - (k - len(row))
                for x in range(current_max + 1, limit + 1):
                    new_diffs = {x - a for a in row}
                    if new_diffs & forbidden_diffs:
                        continue
                    if new_diffs & local_diffs:
                        continue
                    
                    row.append(x)
                    local_diffs_new = local_diffs | new_diffs
                    dfs(row, local_diffs_new)
                    row.pop()
                    
                    # Optimization: If we found a solution, best_last has decreased.
                    # We can break early if we found a solution with last element equal to the theoretical minimum?
                    # Theoretical minimum for a Golomb ruler of length k+1 is roughly k^2/2.
                    # We can't easily know the minimum here.
                    
            dfs([0], set())
            return best_row

    # Main loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(rows)
        
        for i in range(n_rows):
            if time.time() > deadline:
                break
            
            current_row = rows[i]
            current_max = current_row[-1]
            
            # Get forbidden diffs from other rows
            forbidden = get_used_diffs_except_row(rows, i)
            
            # Try to find a better row
            # We want a row with max < current_max
            # Pass current_max as the initial best_last to solve_row
            # Modify solve_row to accept an initial bound
            
            # Let's create a specialized solver call
            # We want to find a row with max < current_max.
            # We can set best_last = current_max in the solver.
            
            # To avoid modifying the nested function, let's just call a helper.
            
            new_row = None
            # We can use a simple greedy with backtracking limited by current_max
            # Let's implement a quick greedy first to see if it helps
            # Greedy for single row:
            greedy_row = [0]
            greedy_diffs = set()
            valid_greedy = True
            for _ in range(k):
                found = False
                # Start from last + 1
                # We want to find smallest x such that diffs are valid
                # and x < current_max (if we want improvement)
                # Actually, we just want the smallest valid x.
                # If the resulting last element is < current_max, we use it.
                
                last_val = greedy_row[-1]
                # We can search up to current_max
                for x in range(last_val + 1, current_max): # strictly less than current_max to ensure improvement?
                    # If we find a solution with last element == current_max, it's not an improvement.
                    # So we should search for x such that the final element < current_max.
                    # This is hard to enforce mid-way.
                    # Better: Find the minimal row. If minimal row's max < current_max, update.
                    
                    new_diffs = {x - a for a in greedy_row}
                    if new_diffs & forbidden or new_diffs & greedy_diffs:
                        continue
                    greedy_row.append(x)
                    greedy_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    valid_greedy = False
                    break
            
            if valid_greedy and greedy_row[-1] < current_max:
                rows[i] = greedy_row
                improved = True
                continue
            
            # If greedy didn't improve, try backtracking for a better solution
            # This is expensive. Only do it if we have time.
            if time.time() < deadline - 5.0:
                # Use the DFS solver
                # We need to pass current_max as the bound
                # Let's refactor solve_row to take a bound
                
                # Re-implementing the solver inline for clarity and speed
                best_row = None
                best_last = current_max # We want < current_max
                
                def dfs(row, local_diffs):
                    nonlocal best_row, best_last
                    if len(row) == k + 1:
                        if row[-1] < best_last:
                            best_row = row[:]
                            best_last = row[-1]
                        return
                    
                    current_max_val = row[-1]
                    if current_max_val + (k + 1 - len(row)) >= best_last:
                        return
                    
                    limit = best_last - (k - len(row))
                    for x in range(current_max_val + 1, limit + 1):
                        new_diffs = {x - a for a in row}
                        if new_diffs & forbidden:
                            continue
                        if new_diffs & local_diffs:
                            continue
                        
                        row.append(x)
                        dfs(row, local_diffs | new_diffs)
                        row.pop()
                        
                        # Early stop if we found the theoretical minimum? 
                        # Or if best_last is very small?
                        # For now, just let it run.
                        
                dfs([0], set())
                
                if best_row:
                    rows[i] = best_row
                    improved = True

    # Return the best rows found
    # We should return the rows with the smallest max
    # The loop updates 'rows' in place, so 'rows' holds the best found so far.
    # However, if an improvement fails, we keep the old row.
    # So 'rows' is always valid.
    
    return rows
