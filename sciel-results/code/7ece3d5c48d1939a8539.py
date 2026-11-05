import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate initial greedy solution
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
    
    current_best_max = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    
    # 2. Backtracking search to improve
    # We want to find a solution with max < current_best_max
    # We can try to find any solution with max <= limit, starting from limit = current_best_max - 1
    
    def solve(row_idx, limit):
        if time.time() > deadline:
            return False
        if row_idx == n_rows:
            return True
        
        # Try to construct row row_idx
        # We need to choose k+1 numbers: 0 = a_0 < a_1 < ... < a_k <= limit
        # such that differences are distinct and not in used_diffs
        
        def construct_row(pos, current_row, row_diffs):
            if time.time() > deadline:
                return False
            
            if pos == k + 1:
                # All differences are in row_diffs
                # Check if row_diffs intersects used_diffs? 
                # We check incrementally, so this should be safe if we updated used_diffs correctly.
                # But we haven't added to global used_diffs yet.
                # So we just need to ensure row_diffs is disjoint from global used_diffs.
                # Wait, the incremental check `not (new_diffs & used_diffs)` ensures this.
                # So if we reach here, the row is valid.
                used_diffs.update(row_diffs)
                rows_temp.append(current_row)
                
                if solve(row_idx + 1, limit):
                    return True
                
                rows_temp.pop()
                for d in row_diffs:
                    used_diffs.discard(d)
                return False
            
            last_val = current_row[-1]
            # Next value must be > last_val and <= limit
            # Also, we must be able to place remaining k-pos elements
            # Minimum possible next value is last_val + 1
            # Maximum possible next value is limit - (k - pos) # at least 1 gap for each remaining
            
            min_next = last_val + 1
            max_next = limit - (k - pos)
            
            # Heuristic: try smaller values first to keep scope small
            # But maybe we should try values that allow more room?
            # Let's iterate from min_next to max_next
            
            for x in range(min_next, max_next + 1):
                new_diffs = {x - a for a in current_row}
                
                # Check against global used
                if new_diffs & used_diffs:
                    continue
                # Check against current row diffs
                if new_diffs & row_diffs:
                    continue
                
                # Valid
                current_row.append(x)
                row_diffs.update(new_diffs)
                
                if construct_row(pos + 1, current_row, row_diffs):
                    return True
                
                current_row.pop()
                for d in new_diffs:
                    row_diffs.discard(d)
            
            return False
        
        # Initialize for this row
        rows_temp = [] # This should be shared or passed? 
        # Actually, `rows` in the outer scope is modified. 
        # It's better to use a local list for the current path and commit at the end.
        # But `solve` is recursive. 
        # Let's pass the list of rows so far.
        
        # Re-structuring: 
        # The `solve` function above is a bit messy with state.
        # Let's use a class or closure with explicit state.
        
        pass # Placeholder

    # The backtracking implementation above is complex to manage state correctly in a closure without errors.
    # Let's implement a cleaner backtracker.
    
    used_diffs = set()
    current_rows = []
    
    def bt(row_idx, limit):
        if time.time() > deadline:
            return False
        if row_idx == n_rows:
            return True
            
        # Construct row row_idx
        # We will use a helper that builds the row and manages backtracking for the row elements
        # and calls bt for the next row.
        
        # To optimize, we can sort the remaining rows? No, they are symmetric.
        
        # We try to build the row.
        # We can use the greedy construction as a hint? 
        # No, just backtrack.
        
        # Optimization: If the limit is small, the search space is small.
        
        # We'll implement the row construction inline or as a nested function.
        
        def build_row(pos, cur_row, cur_diffs):
            if time.time() > deadline:
                return False
            if pos == k + 1:
                # Row is complete and valid
                used_diffs.update(cur_diffs)
                current_rows.append(cur_row)
                
                if bt(row_idx + 1, limit):
                    return True
                
                current_rows.pop()
                for d in cur_diffs:
                    used_diffs.discard(d)
                return False
            
            last = cur_row[-1]
            min_next = last + 1
            # We need to leave room for k - pos elements
            # If we pick x, the next min is x+1.
            # The last element will be at least x + (k - pos).
            # So x + (k - pos) <= limit => x <= limit - (k - pos)
            max_next = limit - (k - pos)
            
            # Heuristic: Try values that are "dense" or "sparse"?
            # Usually, smaller values are better for minimizing max.
            
            for x in range(min_next, max_next + 1):
                new_diffs = {x - a for a in cur_row}
                if new_diffs & used_diffs:
                    continue
                if new_diffs & cur_diffs:
                    continue
                
                cur_row.append(x)
                cur_diffs.update(new_diffs)
                
                if build_row(pos + 1, cur_row, cur_diffs):
                    return True
                
                cur_row.pop()
                for d in new_diffs:
                    cur_diffs.discard(d)
            
            return False
        
        if build_row(1, [0], set()):
            return True
        return False

    # Try to find a solution with limit < current_best_max
    # We can iterate limit downwards from current_best_max - 1
    # Or just try to improve the greedy solution directly using local search if backtracking is too slow.
    
    # Given the constraints (7,5), backtracking with pruning should be very fast.
    # Let's try limits from current_best_max - 1 down to the theoretical minimum.
    
    theoretical_min = (n_rows * k * (k + 1) // 2) # This is count of diffs, so max diff >= count.
    # Actually max diff M satisfies M >= total_diffs.
    # Total diffs = n * k(k+1)/2.
    
    # Let's try to find the minimal limit.
    # We can do a binary search on the limit? 
    # Or just decrement.
    
    # Start from current_best_max - 1
    found_improvement = False
    for limit in range(current_best_max - 1, theoretical_min - 1, -1):
        if time.time() > deadline:
            break
        # Reset state
        used_diffs.clear()
        current_rows.clear()
        
        if bt(0, limit):
            # Found a valid solution with this limit
            # Update best
            best_rows = [r[:] for r in current_rows]
            current_best_max = limit
            found_improvement = True
            # We can continue to search for even smaller limits
            # But we must be careful with time.
            # If we found one, we can try smaller.
            # However, if we found one at L, we know a solution exists for L.
            # We want the minimum L.
            # So we continue the loop to try L-1.
    
    if not found_improvement:
        # If no improvement found via backtracking, return the greedy solution
        return best_rows
    else:
        return best_rows

# Wait, the logic above resets `used_diffs` and `current_rows` inside the loop, which is correct.
# But `bt` uses `used_diffs` and `current_rows` from the closure.
# This should work.

# One issue: The greedy solution might not be the best starting point for the bound.
# But it provides a valid upper bound.

# Let's add a local search step if backtracking doesn't find the absolute minimum quickly, 
# or just rely on backtracking. For small N, backtracking is exact.

# Let's refine the backtracking to be more robust.
# The `build_row` function iterates `x` from `min_next` to `max_next`.
# This is O(Limit * K) per row, which is fine.

# Potential optimization: 
# If `n_rows` is large, backtracking might be slow. 
# But the task specifies "7 rows of 6 integers" in the description, implying the test case is likely small.
# The interface takes `n_rows` and `k`. 
# If `n_rows` is large (e.g., 100), backtracking will fail.
# We should check the size. If `n_rows * k` is large, use local search.

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    # Greedy Initialization
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
    
    current_best_max = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    
    # If problem is small, use exact backtracking
    # Heuristic: n_rows * k < 50?
    if n_rows * k <= 50:
        used_diffs = set()
        current_rows = []
        
        def bt(row_idx, limit):
            if time.time() > deadline:
                return False
            if row_idx == n_rows:
                return True
                
            def build_row(pos, cur_row, cur_diffs):
                if time.time() > deadline:
                    return False
                if pos == k + 1:
                    used_diffs.update(cur_diffs)
                    current_rows.append(cur_row)
                    
                    if bt(row_idx + 1, limit):
                        return True
                    
                    current_rows.pop()
                    for d in cur_diffs:
                        used_diffs.discard(d)
                    return False
                
                last = cur_row[-1]
                min_next = last + 1
                max_next = limit - (k - pos)
                
                for x in range(min_next, max_next + 1):
                    new_diffs = {x - a for a in cur_row}
                    if new_diffs & used_diffs:
                        continue
                    if new_diffs & cur_diffs:
                        continue
                    
                    cur_row.append(x)
                    cur_diffs.update(new_diffs)
                    
                    if build_row(pos + 1, cur_row, cur_diffs):
                        return True
                    
                    cur_row.pop()
                    for d in new_diffs:
                        cur_diffs.discard(d)
                
                return False
            
            if build_row(1, [0], set()):
                return True
            return False

        theoretical_min = n_rows * k * (k + 1) // 2
        # Try to find minimal limit
        # We can binary search or linear search.
        # Linear search from current_best_max - 1 down is safe.
        for limit in range(current_best_max - 1, theoretical_min - 1, -1):
            if time.time() > deadline:
                break
            used_diffs.clear()
            current_rows.clear()
            
            if bt(0, limit):
                best_rows = [r[:] for r in current_rows]
                current_best_max = limit
                # Continue to find even smaller
            else:
                # If no solution for this limit, maybe try smaller? 
                # No, if no solution for L, there is no solution for L-1? 
                # Actually, if no solution for L, there might be a solution for L+1 (which we have).
                # We are searching for the MINIMUM L.
                # If bt(limit) fails, we can't conclude bt(limit-1) fails. 
                # Wait, if a solution exists with max <= L-1, it also exists with max <= L.
                # So if bt(L) fails, it means no solution with max <= L.
                # Thus, no solution with max <= L-1.
                # So we can break? 
                # No, we are iterating DOWN from current_best_max - 1.
                # If bt(L) fails, it means the minimal max is > L.
                # Since we started from current_best_max - 1, and it failed, 
                # it means current_best_max is the minimum? 
                # Not necessarily. We might have skipped a valid configuration?
                # No, bt is exhaustive for a given limit.
                # So if bt(L) fails, the minimal max is > L.
                # So we can break the loop? 
                # Yes, because we are decreasing L. If L fails, L-1 will also fail.
                break
    
    return best_rows

# Let's double check the logic.
# We have a valid solution with max = M_greedy.
# We try limit = M_greedy - 1.
# If bt(M_greedy - 1) succeeds, we have a better solution. We update best.
# We then try limit = M_greedy - 2.
# If bt(M_greedy - 2) fails, it means no solution with max <= M_greedy - 2.
# So the minimal max is M_greedy - 1 (or possibly higher, but we found M_greedy - 1).
# So we can break.
# Correct.

# What if bt(M_greedy - 1) fails immediately?
# Then minimal max is M_greedy. We break and return greedy.

# This seems correct and efficient for small inputs.

# For large inputs, we return the greedy solution.
# Can we improve the greedy solution for large inputs?
# Local search could help.
# But given the time constraint and the specific nature of the task (likely small test cases based on the "7 rows" hint),
# the backtracking approach is the strongest.

# Let's add a small local search improvement for the greedy solution before backtracking, 
# to get a tighter upper bound for the backtracking.

def local_search_improve(rows, n_rows, k, time_limit):
    # Convert to list of lists
    rows = [r[:] for r in rows]
    used = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                used.add(r[j] - r[i])
                
    max_val = max(max(r) for r in rows)
    
    def get_diffs(row, idx):
        # Returns set of diffs involving row[idx]
        # row[0] is 0
        diffs = set()
        val = row[idx]
        for i in range(len(row)):
            if i != idx:
                diffs.add(abs(row[i] - val))
        return diffs

    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try to reduce each element
        for r_idx in range(n_rows):
            row = rows[r_idx]
            for j in range(1, len(row)): # Skip 0
                old_val = row[j]
                # Try to reduce to old_val - 1, - 2, etc?
                # Or just try old_val - 1 first.
                # If it fails, maybe try a larger jump?
                # Let's try to find the largest valid v < old_val.
                # But we want to minimize max, so reducing any element helps if it's the max.
                # If it's not the max, reducing it might allow others to be reduced?
                
                # Simple move: try v = old_val - 1
                new_val = old_val - 1
                if new_val <= row[j-1]:
                    continue
                    
                new_diffs = set()
                for i in range(len(row)):
                    if i != j:
                        new_diffs.add(abs(row[i] - new_val))
                        
                old_diffs = get_diffs(row, j)
                
                # Check conflicts
                # Remove old diffs from used, check new diffs against remaining used
                # Add new diffs, restore old diffs
                # This is expensive. 
                
                # Faster check:
                # new_diffs must not intersect (used - old_diffs)
                # Also new_diffs must be distinct within the row? 
                # The row is a Golomb ruler, so diffs within a row are distinct.
                # Changing one element might create duplicate diffs within the row.
                # So we must check new_diffs against each other? No, they are a set.
                # But we must check if any new_diff is equal to an existing diff in the row (other than those involving j).
                # Let's compute all diffs of the new row.
                
                new_row = row[:]
                new_row[j] = new_val
                
                # Check if new_row is a valid Golomb ruler
                row_diffs = set()
                valid = True
                for i in range(len(new_row)):
                    for l in range(i+1, len(new_row)):
                        d = new_row[l] - new_row[i]
                        if d in row_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check global conflicts
                # Remove old_diffs from used
                # Check if row_diffs (new) intersects used
                # Note: row_diffs includes diffs involving j and not involving j.
                # The diffs not involving j are unchanged.
                # So we only need to check the new diffs involving j against used - old_diffs.
                
                new_j_diffs = {abs(new_val - row[i]) for i in range(len(row)) if i != j}
                
                # Check if new_j_diffs intersects used - old_diffs
                conflict = False
                for d in new_j_diffs:
                    if d in used and d not in old_diffs:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Valid move!
                # Update used
                used -= old_diffs
                used |= new_j_diffs
                
                row[j] = new_val
                max_val = max(max_val, max(row)) # Actually max_val might decrease
                max_val = max(max(r) for r in rows)
                improved = True
                # Break inner loops to restart? Or continue?
                # Continuing might lead to invalid states if we don't update carefully.
                # But we updated `used` and `row`.
                # It's safer to restart the scan or just continue.
                # Let's continue.
                
    return rows

# Integrate local search into construct
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Greedy
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
    
    current_best_max = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    
    # 2. Local Search to improve greedy
    # Time budget: 10s
    ls_deadline = start_time + 10
    improved_rows = local_search_improve(rows, n_rows, k, ls_deadline)
    improved_max = max(max(r) for r in improved_rows)
    if improved_max < current_best_max:
        current_best_max = improved_max
        best_rows = improved_rows
        
    # 3. Backtracking for small problems
    if n_rows * k <= 50:
        used_diffs = set()
        current_rows = []
        
        def bt(row_idx, limit):
            if time.time() > deadline:
                return False
            if row_idx == n_rows:
                return True
                
            def build_row(pos, cur_row, cur_diffs):
                if time.time() > deadline:
                    return False
                if pos == k + 1:
                    used_diffs.update(cur_diffs)
                    current_rows.append(cur_row)
                    
                    if bt(row_idx + 1, limit):
                        return True
                    
                    current_rows.pop()
                    for d in cur_diffs:
                        used_diffs.discard(d)
                    return False
                
                last = cur_row[-1]
                min_next = last + 1
                max_next = limit - (k - pos)
                
                for x in range(min_next, max_next + 1):
                    new_diffs = {x - a for a in cur_row}
                    if new_diffs & used_diffs:
                        continue
                    if new_diffs & cur_diffs:
                        continue
                    
                    cur_row.append(x)
                    cur_diffs.update(new_diffs)
                    
                    if build_row(pos + 1, cur_row, cur_diffs):
                        return True
                    
                    cur_row.pop()
                    for d in new_diffs:
                        cur_diffs.discard(d)
                
                return False
            
            if build_row(1, [0], set()):
                return True
            return False

        theoretical_min = n_rows * k * (k + 1) // 2
        for limit in range(current_best_max - 1, theoretical_min - 1, -1):
            if time.time() > deadline:
                break
            used_diffs.clear()
            current_rows.clear()
            
            if bt(0, limit):
                best_rows = [r[:] for r in current_rows]
                current_best_max = limit
            else:
                break
    
    return best_rows

# The local search `local_search_improve` is O(N * K * K * Time) which might be slow if not optimized.
# The `get_diffs` and conflict check are O(K).
# The loop runs until no improvement.
# For small K, this is fine.

# One detail: `local_search_improve` modifies `rows` in place? 
# No, it takes `rows` and returns `improved_rows`. 
# But inside, it modifies `row` which is a reference to an element of `rows`.
# So `rows` is modified. 
# I should pass a copy to `local_search_improve` to keep the original greedy intact if it fails?
# Actually, if it improves, we want the improved one.
# If it doesn't improve, it returns the same rows (modified but same values? No, it only modifies if valid).
# So it's safe.

# Let's ensure `local_search_improve` doesn't crash.
# `new_val <= row[j-1]` check is good.

# Final check on the backtracking logic.
# `theoretical_min` is the count of differences.
# The max difference must be at least the count.
# So the range is correct.

# The code looks solid.
