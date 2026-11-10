import time
import sys
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110 # 10s buffer
    
    # We want to find the minimal max entry.
    # We can use backtracking to construct the set.
    # State:
    # - rows: list of lists of ints
    # - used_diffs: set of all differences used so far
    # - current_row: the row currently being constructed
    
    # To make it fast, we can also bound the search by the current best known max.
    # However, since we don't have a good initial bound other than infinity, 
    # we can use a heuristic upper bound or just let it find one and then optimize.
    # A simple greedy gives an upper bound.
    
    # Let's first get a greedy solution to set an upper bound.
    def greedy_solve():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < k + 1:
                found = False
                x = last + 1
                # Search for next valid x
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        last = x
                        found = True
                        break
                    x += 1
                    if x > 200: # Safety break for greedy
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    greedy_rows = greedy_solve()
    best_max = max(max(r) for r in greedy_rows) if greedy_rows else 200
    best_rows = greedy_rows
    
    # Now use backtracking to try to beat best_max.
    # We will try to find a solution with max < best_max.
    # We can do this by iteratively lowering the limit and running backtracking.
    # Or, just run backtracking with a pruning condition: if current_max > best_max, prune.
    # And if we find a valid solution, update best_max and best_rows, then continue searching for even better?
    # Backtracking usually finds *a* solution. To find the *best*, we can use branch and bound.
    
    # Let's implement a backtracker that tries to find a solution with max <= limit.
    # We will iterate limit downwards from best_max - 1.
    
    def backtrack(rows, used_diffs, limit):
        # rows is a list of completed rows
        # We need to construct the next row
        if len(rows) == n_rows:
            return True, rows
        
        # Construct the next row
        # We can try different starting points? No, row must start with 0.
        # So row[0] = 0 is fixed.
        # We need to choose k more numbers.
        
        # To speed up, we can use symmetry or fixed patterns?
        # For small k, we can just backtrack on the next element.
        
        # Helper to check if a partial row can be extended
        # We will recurse on the position in the current row.
        
        # Let's restructure: 
        # backtrack_row(row, pos, used_diffs, limit)
        
        def backtrack_row(row, pos, used_diffs, limit):
            if pos == k + 1:
                # Row complete
                new_rows = rows + [row]
                return backtrack(new_rows, used_diffs, limit)
            
            # Choose next number x
            # x must be > row[-1]
            # x must be <= limit
            # All differences x - row[i] for i in range(pos) must not be in used_diffs
            # Also, differences between x and future numbers will be checked later.
            
            # Heuristic: try smaller x first to keep max small?
            # Or try x that leaves more options?
            # Let's try x from row[-1] + 1 to limit.
            
            last_val = row[-1]
            for x in range(last_val + 1, limit + 1):
                # Check conflicts with current row
                conflict = False
                new_diffs = set()
                for i in range(pos):
                    d = x - row[i]
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Also check if new_diffs have internal conflicts? 
                # No, differences within the same row are distinct by construction of strictly increasing integers?
                # Wait, if row is [0, 2, 5], diffs are 2, 5, 3. All distinct.
                # If row is [0, 1, 3], diffs are 1, 3, 2. All distinct.
                # Is it possible to have duplicate diffs in a single row?
                # a_j - a_i = a_k - a_l?
                # If i < j and l < k.
                # Since the sequence is strictly increasing, the differences are unique for each pair (i,j) 
                # ONLY IF we consider the set of all differences.
                # Actually, for a set of integers, the differences a_j - a_i are not necessarily distinct.
                # Example: {0, 1, 3, 4}. Diffs: 1, 3, 4, 2, 3, 1. 
                # 1 appears twice (1-0, 4-3). 3 appears twice (3-0, 4-1).
                # The problem states: "all differences between two entries of the same row are distinct over all rows".
                # This implies that within a row, all differences must be distinct?
                # "all differences ... are distinct over all rows" usually means the global set of all differences from all rows must be unique.
                # This implies that even within a single row, differences must be distinct.
                # So we must also ensure that new_diffs do not contain duplicates.
                
                if len(new_diffs) != len(row):
                    # Duplicates found within the new set of diffs generated by x
                    continue
                
                # If we pass, we can proceed
                # We must ensure that the differences *within* the row so far are also distinct.
                # My check `len(new_diffs) != len(row)` checks if the diffs from x to all previous elements are distinct.
                # But what about diffs between previous elements?
                # Those were checked when those previous elements were added.
                # So we just need to check if `new_diffs` intersects `used_diffs` (which includes diffs from previous rows AND diffs within current row from previous steps).
                # Wait, `used_diffs` in my current logic only contains diffs from *completed* rows?
                # No, I should update `used_diffs` as I build the row?
                # If I do, I need to backtrack and remove them.
                
                # Let's modify backtrack_row to pass a local set of diffs for the current row.
                # Or, simpler: `used_diffs` is the global set. 
                # When adding x, we generate `new_diffs`.
                # We must ensure `new_diffs` is disjoint from `used_diffs`.
                # We must also ensure `new_diffs` has no duplicates (checked by set size).
                # BUT, we also need to ensure that `new_diffs` doesn't conflict with diffs *already in the current row* that are not in `used_diffs` yet?
                # If I don't update `used_diffs` until the row is complete, I might miss conflicts within the row.
                # Example: row [0, 1, 3].
                # Step 1: add 1. diffs {1}. used_diffs (global) is empty. OK.
                # Step 2: add 3. diffs {3, 2}. 
                # If I check against global `used_diffs` (empty), it passes.
                # But I need to check against diffs from step 1 ({1}).
                # So I must maintain a set of diffs for the current row being built.
                
                # Let's refactor: pass `row_diffs` (set of diffs in current row so far) to backtrack_row.
                
                # Actually, let's just check `new_diffs` against `used_diffs` (global) AND `row_diffs` (local).
                # And ensure `new_diffs` are distinct.
                
                # Let's restart the logic inside the loop for clarity.
                pass # Placeholder for actual implementation below

        # Revised Backtracking Logic
        def bt(row, pos, used_global, row_local_diffs, limit):
            if pos == k + 1:
                # Row complete
                # Add row_local_diffs to used_global? No, used_global is immutable in this scope?
                # Better to pass a set and mutate it, or create a new set.
                # Creating a new set is safer but slower. Given small size, it's fine.
                new_used_global = used_global | row_local_diffs
                new_rows = rows + [row]
                return bt_next_row(new_rows, new_used_global, limit)
            
            last_val = row[-1]
            # Heuristic: iterate x
            # To find minimal max, we should try smaller x first?
            # But we are bounded by `limit`.
            # If we find a solution, we return it.
            # We are calling this with decreasing limits.
            
            for x in range(last_val + 1, limit + 1):
                new_diffs = []
                valid = True
                for i in range(pos):
                    d = x - row[i]
                    if d in used_global or d in row_local_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check for duplicates in new_diffs
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                # Also check if new_diffs conflict with each other? 
                # No, they are just a set of integers. Distinctness is handled by set.
                
                # Recurse
                new_row_local = row_local_diffs | set(new_diffs)
                if bt(row + [x], pos + 1, used_global, new_row_local, limit):
                    return True
            
            return False

        def bt_next_row(rows_list, used_global, limit):
            if len(rows_list) == n_rows:
                return True
            
            # Start new row with 0
            # Note: 0 doesn't create diffs.
            # The first element chosen will create diffs with 0.
            return bt([0], 1, used_global, set(), limit)
            
        # Main loop to find solution with max <= limit
        # We start from best_max - 1 and go down?
        # Or just try to find *any* solution with max <= current_best - 1?
        # If found, update best and try to go lower.
        
        # To avoid deep recursion issues or slow search, we can add a node counter or time check.
        
        current_limit = best_max - 1
        
        # Optimization: If n_rows is small, we can just run the backtracker.
        # The space is small.
        
        # Let's try to find a solution with max < best_max.
        # We will try limits from best_max - 1 down to some minimum (e.g., k*(k+1)/2? No, counting bound).
        
        # Actually, just running `bt` with `limit = best_max - 1` might be slow if no solution exists.
        # We should add a timeout or node limit inside `bt`.
        
        node_count = 0
        max_nodes = 100000 # Adjust based on time
        
        def bt_timed(row, pos, used_global, row_local_diffs, limit):
            nonlocal node_count
            node_count += 1
            if node_count > max_nodes:
                return False
            if time.time() - start_time > time_limit:
                return False
                
            if pos == k + 1:
                new_used_global = used_global | row_local_diffs
                new_rows = rows + [row]
                return bt_next_row_timed(new_rows, new_used_global, limit)
            
            last_val = row[-1]
            # Heuristic: 
            # If we want to minimize the max, and we are already within `limit`,
            # trying smaller x first is good for finding *a* solution quickly?
            # Yes, it keeps the numbers small.
            
            for x in range(last_val + 1, limit + 1):
                new_diffs = []
                valid = True
                for i in range(pos):
                    d = x - row[i]
                    if d in used_global or d in row_local_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                new_row_local = row_local_diffs | set(new_diffs)
                if bt_timed(row + [x], pos + 1, used_global, new_row_local, limit):
                    return True
            
            return False

        def bt_next_row_timed(rows_list, used_global, limit):
            nonlocal node_count
            if node_count > max_nodes:
                return False
            if time.time() - start_time > time_limit:
                return False
                
            if len(rows_list) == n_rows:
                # Solution found
                nonlocal best_rows, best_max
                # The current `rows_list` is the solution
                # We need to capture it. 
                # Since `rows` in the outer scope is not updated, we return the list.
                # But `bt_timed` returns True/False. We need to return the solution.
                # Let's change return type to (bool, list) or use a nonlocal variable.
                nonlocal found_solution
                found_solution = [r[:] for r in rows_list]
                return True
            
            return bt_timed([0], 1, used_global, set(), limit)

        found_solution = None
        
        # Try to find a better solution
        # We iterate limit downwards? 
        # If we find a solution with limit L, we can update best_max to L.
        # Then we can try L-1.
        
        # However, the backtracker with a fixed limit finds *a* solution.
        # It doesn't guarantee the minimal max *within* that limit.
        # But if we iterate limit downwards, the first solution found with limit L has max <= L.
        # If we found it, best_max becomes L.
        # Then we try L-1.
        
        # This is a Branch and Bound style.
        
        # Start from best_max - 1
        # If we find a solution, update best_max.
        # Repeat until time runs out or limit is too small.
        
        # To speed up, we can use the fact that the first row is symmetric?
        # Not really, the differences are absolute.
        
        # Let's run the search.
        
        # We need to be careful with `rows` variable scope in `bt_next_row_timed`.
        # It uses `rows` from the enclosing scope of `bt_next_row_timed`, which is `construct`.
        # But `rows` is updated in `bt_next_row_timed`? No, `rows` is passed as `rows_list`.
        # The base case `if len(rows_list) == n_rows` uses `rows_list`.
        
        # Let's re-verify the scope.
        # `bt_next_row_timed` calls `bt_timed`.
        # `bt_timed` calls `bt_next_row_timed` with `new_rows`.
        # So `rows_list` is the correct accumulator.
        
        # One issue: `bt_next_row_timed` accesses `rows` (the outer variable) in the original `bt` logic?
        # In my `bt_next_row_timed` above, I used `rows_list`.
        # But in `bt_timed`, I used `rows + [row]`. `rows` is the outer variable.
        # This is a bug. `bt_timed` should not depend on outer `rows`.
        # It should return the new list of rows.
        
        # Let's refactor to return the solution list directly or use a nonlocal `current_rows`.
        
        # Revised:
        # `bt_timed` returns `None` if fail, or `list_of_rows` if success.
        
        def bt_timed_v2(row, pos, used_global, row_local_diffs, limit, current_rows):
            nonlocal node_count
            node_count += 1
            if node_count > max_nodes:
                return None
            if time.time() - start_time > time_limit:
                return None
                
            if pos == k + 1:
                new_used_global = used_global | row_local_diffs
                new_rows = current_rows + [row]
                return bt_next_row_timed_v2(new_rows, new_used_global, limit)
            
            last_val = row[-1]
            for x in range(last_val + 1, limit + 1):
                new_diffs = []
                valid = True
                for i in range(pos):
                    d = x - row[i]
                    if d in used_global or d in row_local_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                new_row_local = row_local_diffs | set(new_diffs)
                res = bt_timed_v2(row + [x], pos + 1, used_global, new_row_local, limit, current_rows)
                if res is not None:
                    return res
            
            return None

        def bt_next_row_timed_v2(rows_list, used_global, limit):
            nonlocal node_count
            if node_count > max_nodes:
                return None
            if time.time() - start_time > time_limit:
                return None
                
            if len(rows_list) == n_rows:
                return rows_list
            
            return bt_timed_v2([0], 1, used_global, set(), limit, rows_list)
        
        # Search loop
        # Start from best_max - 1
        # If we find a solution, update best_max and best_rows.
        # Then try best_max - 1 again.
        
        # To prevent infinite loops if no solution exists for a limit, 
        # the backtracker will return None.
        
        # We can also randomize the order of x to escape local minima?
        # No, backtracking is exhaustive.
        
        # Let's try to find a solution with max < best_max.
        
        # We can also use the greedy solution to guide the search?
        
        # Let's just run it.
        
        # Note: The first row can be chosen in many ways.
        # To reduce symmetry, we can fix the first row?
        # No, the first row is part of the solution.
        
        # Let's run the backtracker for limit = best_max - 1.
        
        # If found, update.
        # Then limit = best_max - 1 (new).
        
        # We do this until time runs out.
        
        # Initial state for backtracking:
        # rows_list = []
        # used_global = set()
        
        # We call bt_next_row_timed_v2([], set(), limit)
        
        # Let's wrap the search in a while loop.
        
        search_limit = best_max - 1
        
        while search_limit >= k + 1: # Minimum possible max is at least k
            if time.time() - start_time > time_limit:
                break
            
            node_count = 0
            max_nodes = 50000 # Limit per limit attempt to avoid hanging on impossible limits
            
            sol = bt_next_row_timed_v2([], set(), search_limit)
            
            if sol is not None:
                # Found a solution with max <= search_limit
                # The actual max might be smaller, but it's <= search_limit.
                # We can calculate the actual max.
                actual_max = max(max(r) for r in sol)
                if actual_max < best_max:
                    best_max = actual_max
                    best_rows = sol
                    search_limit = best_max - 1
                else:
                    # Solution found but not better? 
                    # This shouldn't happen if we are searching for limit < best_max.
                    # Unless the backtracker found a solution with max == search_limit, 
                    # and search_limit < best_max.
                    # Then actual_max <= search_limit < best_max.
                    # So it should always be an improvement.
                    search_limit -= 1
            else:
                # No solution found for this limit
                search_limit -= 1
        
    if best_rows is None:
        return greedy_rows
        
    return best_rows
