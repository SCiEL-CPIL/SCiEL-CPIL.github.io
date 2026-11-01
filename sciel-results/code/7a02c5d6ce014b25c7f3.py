import random
import math
import time

def construct(n_rows, k):
    # Target: n_rows rows, each of length k+1
    # Start with a greedy construction to have a valid starting point
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Get initial solution
    rows = greedy_construct()
    
    # Helper to calculate the set of differences for a single row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to find a valid row with minimal max value given forbidden diffs
    # Uses a greedy/backtracking hybrid. 
    # Since k is small (5), the row length is 6.
    # We can use a simple DFS with pruning.
    def find_best_row(forbidden_diffs, current_max_limit=None):
        # We want to find a row [0, a1, ..., ak] such that all diffs are not in forbidden_diffs
        # and all diffs are distinct within the row (implied by distinctness if we check against global? 
        # No, diffs within the row must be distinct from each other AND from forbidden_diffs).
        # So the set of diffs of the new row must be disjoint from forbidden_diffs.
        # And the set of diffs of the new row must have size (k+1)*k//2.
        
        best_row = None
        best_max = float('inf')
        
        # To speed up, we can iterate possible max values or use a greedy approach with randomization
        # Let's use a recursive search with random ordering for ties to allow multiple attempts if needed.
        # But for now, a deterministic greedy that picks the smallest valid next element is a good start.
        # To allow local search, we might want to find *a* valid row, not necessarily the absolute best one, 
        # or the best one under a constraint.
        
        # Let's try to construct the row greedily, but with a twist: 
        # We will try to keep the max element as small as possible.
        
        row = [0]
        used_in_row = set()
        
        def backtrack(idx, last_val):
            nonlocal best_row, best_max
            if idx == k + 1:
                if row[-1] < best_max:
                    best_max = row[-1]
                    best_row = row[:]
                return True # Found a valid row, we can stop or continue to find better?
                            # For local search, finding *a* valid row that is better than current is enough.
                            # If we find one, we can return.
            # Try next values starting from last_val + 1
            # Optimization: The max possible value is bounded by current best_max or some heuristic bound
            # Let's just iterate.
            for x in range(last_val + 1, best_max): # Prune if x >= best_max
                # Check if x - a_i for all a_i in row are distinct and not in forbidden_diffs
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden_diffs:
                    continue
                if new_diffs & used_in_row:
                    continue
                if len(new_diffs) != len(row): # Distinctness check within new diffs
                    continue
                
                row.append(x)
                used_in_row.update(new_diffs)
                
                if backtrack(idx + 1, x):
                    # Unwind
                    row.pop()
                    for d in new_diffs:
                        used_in_row.discard(d)
                    return True
                else:
                    row.pop()
                    for d in new_diffs:
                        used_in_row.discard(d)
            
            return False
        
        # We need to be careful with the global state of 'row' and 'used_in_row' in the closure
        # Let's re-implement cleanly to avoid closure issues in recursion
        def dfs(row_so_far, used_diffs_so_far, last_val):
            if len(row_so_far) == k + 1:
                return row_so_far
            for x in range(last_val + 1, best_max):
                new_diffs = {x - a for a in row_so_far}
                if new_diffs & forbidden_diffs:
                    continue
                if new_diffs & used_diffs_so_far:
                    continue
                if len(new_diffs) != len(row_so_far):
                    continue
                
                res = dfs(row_so_far + [x], used_diffs_so_far | new_diffs, x)
                if res:
                    return res
            return None

        # To make it faster, we can set a bound on best_max. 
        # If we are improving a row, we know the current max. 
        # Let's pass the current max as the upper bound for search.
        # If we don't find one smaller, we return None.
        
        return dfs([0], set(), 0)

    # Main Local Search Loop
    # We iterate through rows, trying to improve each one.
    # We use a "temperature" or random factor to sometimes accept worse rows? 
    # For simplicity, let's do greedy improvement first.
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    # Precompute diffs for all rows
    all_diffs = [get_row_diffs(r) for r in rows]
    
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        for i in range(n_rows):
            if (time.time() - start_time) > timeout:
                break
            
            # Calculate forbidden diffs: all diffs from rows except i
            forbidden = set()
            for j in range(n_rows):
                if j != i:
                    forbidden |= all_diffs[j]
            
            current_max = rows[i][-1]
            
            # Try to find a row with max < current_max
            # We set best_max to current_max in the finder
            # The dfs function above uses a non-local best_max which is tricky.
            # Let's rewrite the finder to take a max_limit
            
            def find_row_with_limit(forbidden_diffs, max_limit):
                # Returns a row with max < max_limit, or None
                # Uses DFS
                def dfs(row_so_far, used_diffs_so_far, last_val):
                    if len(row_so_far) == k + 1:
                        if row_so_far[-1] < max_limit:
                            return row_so_far
                        return None
                    # Heuristic: try smaller x first
                    for x in range(last_val + 1, max_limit):
                        new_diffs = {x - a for a in row_so_far}
                        if new_diffs & forbidden_diffs:
                            continue
                        if new_diffs & used_diffs_so_far:
                            continue
                        if len(new_diffs) != len(row_so_far):
                            continue
                        
                        res = dfs(row_so_far + [x], used_diffs_so_far | new_diffs, x)
                        if res:
                            return res
                    return None

                return dfs([0], set(), 0)

            new_row = find_row_with_limit(forbidden, current_max)
            
            if new_row:
                # Update row i
                rows[i] = new_row
                all_diffs[i] = get_row_diffs(new_row)
                improved = True
                # We could restart the loop or continue to next row. 
                # Continuing to next row is fine.
    
    # If time remains, we can try random restarts or perturbations
    # But for small n, the greedy local search is often sufficient to beat the naive seed significantly.
    # The seed is purely greedy forward. This local search refines it.
    
    return rows
