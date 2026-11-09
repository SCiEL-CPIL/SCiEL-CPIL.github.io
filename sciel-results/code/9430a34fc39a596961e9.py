import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    n_rows: number of rows
    k: parameter such that each row has k+1 elements.
    """
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We should return well before that.
    time_limit = 140.0
    
    k_plus_1 = k + 1  # Number of elements per row
    
    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Checks if the rows form a valid difference triangle set."""
        used = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    bit = 1 << diff
                    if used & bit:
                        return False
                    used |= bit
        return True
        
    def compute_used_bits(rows):
        """Computes the bitset of all used differences."""
        used = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    used |= (1 << diff)
        return used

    def find_min_row(forbidden_bits, k, max_limit, time_limit=None):
        """
        Finds a row of length k+1 starting at 0, with strictly increasing integers,
        such that all differences are distinct and not in forbidden_bits.
        Minimizes the last element (the max of the row).
        Returns the row list or None if not found within max_limit or time_limit.
        """
        if time_limit is not None and time.time() > time_limit:
            return None
            
        best_row = None
        best_max = max_limit + 1
        
        # DFS state: current_row, current_max, used_diffs_in_row (bitset)
        # We want to find ANY row with max < best_max.
        
        def dfs(current_row, current_max, used_in_row_bits):
            nonlocal best_row, best_max
            
            if len(current_row) == k_plus_1:
                if current_max < best_max:
                    best_max = current_max
                    best_row = list(current_row)
                return True
            
            # Pruning: if current_max is already >= best_max, no need to continue this branch
            # because the next element will be even larger.
            if current_max >= best_max:
                return False
            
            # Determine the range for the next element x
            # x must be > current_max
            # x must be < best_max (to improve on current best)
            # Also x must be <= max_limit (hard cap)
            
            start_x = current_max + 1
            end_x = min(best_max - 1, max_limit)
            
            if start_x > end_x:
                return False
            
            # Iterate x from start_x to end_x
            # Heuristic: Trying smaller x first helps find tight rows.
            for x in range(start_x, end_x + 1):
                # Calculate new differences
                new_diffs_bits = 0
                valid = True
                for a in current_row:
                    d = x - a
                    # d must be > 0 (guaranteed since x > a)
                    bit = 1 << d
                    # Check if d is in forbidden_bits or used_in_row_bits
                    if (forbidden_bits | used_in_row_bits) & bit:
                        valid = False
                        break
                    new_diffs_bits |= bit
                
                if not valid:
                    continue
                
                # Check if new_diffs_bits has any internal collisions?
                # No, because we are adding x to a set of distinct previous numbers.
                # The differences x-a_i are distinct if a_i are distinct.
                # So we only need to check against forbidden and existing row diffs.
                
                # Recurse
                current_row.append(x)
                if dfs(current_row, x, used_in_row_bits | new_diffs_bits):
                    # Found a better solution
                    current_row.pop()
                    return True
                
                current_row.pop()
            
            return False
        
        # Initial call
        # We don't know the best_max yet, so we start with max_limit
        # To speed up, we can do a quick greedy to find a decent upper bound for best_max first?
        # Or just let DFS handle it. The pruning `current_max >= best_max` is key.
        # Initially best_max is max_limit + 1.
        
        # Optimization: Do a quick greedy to set a good initial best_max
        greedy_row = [0]
        greedy_used = 0
        current_max = 0
        for _ in range(k):
            found = False
            x = current_max + 1
            # Limit search for greedy to max_limit
            while x <= max_limit:
                new_diffs_bits = 0
                valid = True
                for a in greedy_row:
                    d = x - a
                    bit = 1 << d
                    if (forbidden_bits | greedy_used) & bit:
                        valid = False
                        break
                    new_diffs_bits |= bit
                if valid:
                    greedy_row.append(x)
                    greedy_used |= new_diffs_bits
                    current_max = x
                    found = True
                    break
                x += 1
            if not found:
                return None # Greedy failed, likely DFS will be hard too, but let's try DFS anyway?
                # Actually if greedy fails, there might still be a solution with a different structure.
                # But for speed, if greedy fails with a wide search, it's a bad sign.
                # Let's just return None if greedy fails to save time, or let DFS try?
                # Let's let DFS try, but with a tight time limit.
        
        if greedy_row:
            best_max = max(greedy_row)
            best_row = list(greedy_row)
        
        # Now run DFS to find if we can do better than greedy
        # Reset best_row to None to force DFS to find something? 
        # No, if DFS finds nothing better, we return the greedy one.
        # But DFS needs to know the current best to prune.
        
        # We need to be careful: if best_row is from greedy, we want DFS to find < best_max.
        # The DFS logic above returns True if it finds a solution with max < best_max.
        # If it returns False, best_row remains the greedy one.
        
        # However, the DFS function above modifies best_row and best_max.
        # We need to pass the initial best_max from greedy.
        
        # Let's restructure:
        # 1. Greedy to get a candidate.
        # 2. DFS with best_max = greedy_max to see if we can improve.
        
        # Re-implementing DFS to take initial_best_max
        def dfs_improve(current_row, current_max, used_in_row_bits, limit_max):
            nonlocal best_row, best_max
            
            if len(current_row) == k_plus_1:
                if current_max < best_max:
                    best_max = current_max
                    best_row = list(current_row)
                return True
            
            if current_max >= best_max:
                return False
            
            start_x = current_max + 1
            end_x = min(best_max - 1, max_limit)
            
            if start_x > end_x:
                return False
            
            for x in range(start_x, end_x + 1):
                new_diffs_bits = 0
                valid = True
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if (forbidden_bits | used_in_row_bits) & bit:
                        valid = False
                        break
                    new_diffs_bits |= bit
                
                if not valid:
                    continue
                
                current_row.append(x)
                if dfs_improve(current_row, x, used_in_row_bits | new_diffs_bits, limit_max):
                    current_row.pop()
                    return True
                current_row.pop()
            
            return False
        
        # If greedy found a row, try to improve it
        if best_row is not None:
            # We need to reset best_row to None? No, we want to keep it if DFS fails.
            # But DFS modifies best_row. We should save the greedy result.
            greedy_result = best_row
            greedy_max = best_max
            
            # Try DFS to find better
            # We need a fresh DFS state.
            # The function above uses nonlocal best_row. 
            # Let's just call it. If it finds nothing, best_row is still greedy_result?
            # No, if it finds nothing, best_row is unchanged.
            # But wait, if it finds something, it updates it.
            
            # To be safe, let's run DFS with a time limit.
            # The DFS above doesn't have a time check. Let's add one.
            
            def dfs_timed(current_row, current_max, used_in_row_bits):
                nonlocal best_row, best_max
                
                if time.time() > time_limit:
                    return False
                    
                if len(current_row) == k_plus_1:
                    if current_max < best_max:
                        best_max = current_max
                        best_row = list(current_row)
                    return True
                
                if current_max >= best_max:
                    return False
                
                start_x = current_max + 1
                end_x = min(best_max - 1, max_limit)
                
                if start_x > end_x:
                    return False
                
                for x in range(start_x, end_x + 1):
                    new_diffs_bits = 0
                    valid = True
                    for a in current_row:
                        d = x - a
                        bit = 1 << d
                        if (forbidden_bits | used_in_row_bits) & bit:
                            valid = False
                            break
                        new_diffs_bits |= bit
                    
                    if not valid:
                        continue
                    
                    current_row.append(x)
                    if dfs_timed(current_row, x, used_in_row_bits | new_diffs_bits):
                        current_row.pop()
                        return True
                    current_row.pop()
                
                return False

            # Run DFS
            # We need to be careful with the nonlocal variables.
            # Let's just call dfs_timed.
            # But we need to pass the initial state.
            # The function uses best_max which is currently greedy_max.
            # So it will only accept solutions < greedy_max.
            
            # Note: If greedy_max is already very small, DFS might be fast.
            # If greedy_max is large, DFS might be slow.
            
            # To avoid hanging, we rely on the time_limit.
            
            # Call DFS
            # We need a fresh row to start DFS?
            # No, we can start from scratch to find a potentially different structure that is tighter.
            # Or we can try to modify the greedy row?
            # Let's start from scratch [0] to find the absolute minimum for this constraint.
            
            # Save current best
            saved_best_row = best_row
            saved_best_max = best_max
            
            # Run DFS from scratch
            # We need to reset best_max to infinity? 
            # No, we want to find something < saved_best_max.
            # So we set best_max = saved_best_max.
            # And best_row = None? 
            # If DFS finds nothing, we want to return saved_best_row.
            
            # Let's modify the logic:
            # 1. Greedy finds row G with max M_G.
            # 2. Set best_max = M_G, best_row = G.
            # 3. Run DFS. If DFS finds row D with max M_D < M_G, update best.
            # 4. Return best.
            
            # The dfs_timed function above updates best_row/best_max if it finds a better one.
            # It starts with current_row=[0].
            
            # We must ensure time_limit is set.
            if time_limit is not None:
                dfs_timed([0], 0, 0)
            
            # After DFS, best_row contains the best found (either greedy or DFS improvement)
            return best_row
        
        else:
            # Greedy failed. Try DFS from scratch with a large max_limit?
            # If greedy failed, it's likely no solution exists in max_limit.
            # But let's try DFS with a small time limit.
            if time_limit is not None:
                # Set best_max to max_limit + 1 so DFS can find any solution
                best_max = max_limit + 1
                dfs_timed([0], 0, 0)
                if best_row is not None:
                    return best_row
            return None

    def construct_rows_greedy_dfs(seed=0):
        """
        Construct all rows sequentially, using DFS to minimize each row's max
        given the differences used by previous rows.
        """
        random.seed(seed)
        used_bits = 0
        rows = []
        
        for row_idx in range(n_rows):
            # Determine a reasonable max_limit for this row
            # Start with a limit that allows some slack
            # The counting bound is 112. Best known 134.
            # Let's start with 150 and see.
            # Actually, we can try to find the minimal max.
            # Let's set a high limit, e.g., 200, and let DFS minimize it.
            max_limit = 200
            
            # Time limit for this row construction
            # Allocate time based on remaining time
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            if remaining < 5:
                break
                
            time_per_row = min(10.0, remaining / (n_rows - row_idx))
            
            row = find_min_row(used_bits, k, max_limit, time_limit=time.time() + time_per_row)
            
            if row is None:
                # Fallback: simple greedy
                row = [0]
                current_max = 0
                while len(row) < k_plus_1:
                    x = current_max + 1
                    while True:
                        new_diffs_bits = 0
                        valid = True
                        for a in row:
                            d = x - a
                            bit = 1 << d
                            if used_bits & bit:
                                valid = False
                                break
                            new_diffs_bits |= bit
                        if valid:
                            row.append(x)
                            used_bits |= new_diffs_bits
                            current_max = x
                            break
                        x += 1
                # Note: The fallback greedy above updates used_bits inside the loop, 
                # but we need to be careful. 
                # Actually, the loop above updates used_bits as it builds the row.
                # So after the loop, used_bits is correct.
            else:
                # Update used_bits with the new row's differences
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        diff = row[j] - row[i]
                        used_bits |= (1 << diff)
            
            rows.append(row)
            
        return rows

    def local_search(rows, max_time):
        """
        Local search to improve the solution.
        Uses bitsets for speed.
        """
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        used_bits = compute_used_bits(rows)
        
        start = time.time()
        improved = True
        stall_count = 0
        seed_counter = 0
        
        while improved and time.time() - start < max_time:
            improved = False
            stall_count += 1
            
            # Strategy 1: Shrink elements
            # Identify elements to try shrinking
            # We want to reduce the global max.
            # Try shrinking the largest elements first.
            
            elements = []
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if i > 0:
                        elements.append((rows[r][i], r, i))
            elements.sort(reverse=True)
            
            found_improvement = False
            # Try top N elements
            for val, max_r, max_i in elements[:10]:
                if time.time() - start > max_time - 0.1:
                    break
                
                row = rows[max_r]
                current_val = row[max_i]
                if current_val <= 0:
                    continue
                
                min_val = row[max_i - 1] + 1
                max_allowed = current_val - 1
                if max_i < len(row) - 1:
                    max_allowed = min(max_allowed, row[max_i + 1] - 1)
                
                if max_allowed < min_val:
                    continue
                
                # Remove current val's differences from used_bits
                old_diffs_bits = 0
                for a in row:
                    if a != current_val:
                        d = abs(a - current_val)
                        old_diffs_bits |= (1 << d)
                
                # We need to check if these diffs are unique to this element?
                # No, a difference d = |a - b| is shared by pair (a,b).
                # If we change b to b', the diff |a-b| is removed, and |a-b'| is added.
                # But |a-b| might be the same as |c-d| for another pair?
                # No, all differences are distinct globally.
                # So removing the diffs involving current_val is safe.
                
                used_bits_without = used_bits & ~old_diffs_bits
                
                new_val_found = None
                # Try values from min_val to current_val - 1
                # Prefer smaller values? Yes, to reduce max.
                for v in range(min_val, current_val):
                    # Check if v is valid
                    new_diffs_bits = 0
                    valid = True
                    for a in row:
                        if a != current_val:
                            d = abs(a - v)
                            bit = 1 << d
                            if used_bits_without & bit:
                                valid = False
                                break
                            new_diffs_bits |= bit
                    
                    # Also check internal consistency?
                    # The new diffs are between v and other elements in the row.
                    # Since v is distinct from all other elements, the diffs are distinct among themselves.
                    # We just need to ensure they don't conflict with used_bits_without.
                    
                    if valid:
                        new_val_found = v
                        break
                
                if new_val_found is not None:
                    row[max_i] = new_val_found
                    used_bits = used_bits_without | new_diffs_bits
                    found_improvement = True
                    score = get_score(rows)
                    if score < best_score:
                        best_score = score
                        best_rows = [list(r) for r in rows]
                    break
                else:
                    # Restore used_bits
                    used_bits = used_bits_without | old_diffs_bits
            
            if found_improvement:
                improved = True
                continue
            
            # Strategy 2: Rebuild the row with the current global maximum
            max_val = -1
            max_r = -1
            for r in range(n_rows):
                if max(rows[r]) > max_val:
                    max_val = max(rows[r])
                    max_r = r
            
            if max_r != -1 and time.time() - start < max_time - 1.0:
                seed_counter += 1
                # Compute used_bits for other rows
                other_rows = [r for i, r in enumerate(rows) if i != max_r]
                other_used_bits = compute_used_bits(other_rows)
                
                # Find a new row that minimizes max, given other_used_bits
                # Limit the max to current max_val - 1 to force improvement?
                # Or just find the minimal possible.
                # If minimal possible is < max_val, we accept.
                
                # Time limit for this rebuild
                rebuild_time = min(5.0, max_time - (time.time() - start) - 0.5)
                
                new_row = find_min_row(other_used_bits, k, max_val - 1, time_limit=time.time() + rebuild_time)
                
                if new_row is not None and max(new_row) < max_val:
                    rows[max_r] = new_row
                    used_bits = compute_used_bits(rows)
                    score = get_score(rows)
                    if score < best_score:
                        best_score = score
                        best_rows = [list(r) for r in rows]
                    improved = True
                    continue
            
            # Strategy 3: Try rebuilding other rows if stalled
            if stall_count >= 2 and time.time() - start < max_time - 1.0:
                row_order = sorted(range(n_rows), key=lambda r: max(rows[r]), reverse=True)
                for r_to_rebuild in row_order[:2]:
                    if time.time() - start > max_time - 0.5:
                        break
                    seed_counter += 1
                    other_rows = [r for i, r in enumerate(rows) if i != r_to_rebuild]
                    other_used_bits = compute_used_bits(other_rows)
                    
                    current_max = max(rows[r_to_rebuild])
                    rebuild_time = min(3.0, max_time - (time.time() - start) - 0.5)
                    
                    new_row = find_min_row(other_used_bits, k, current_max - 1, time_limit=time.time() + rebuild_time)
                    
                    if new_row is not None and max(new_row) < current_max:
                        rows[r_to_rebuild] = new_row
                        used_bits = compute_used_bits(rows)
                        score = get_score(rows)
                        if score < best_score:
                            best_score = score
                            best_rows = [list(r) for r in rows]
                        improved = True
                        break
            
            if not found_improvement:
                improved = False
                stall_count += 1
                if stall_count > 10:
                    break
        
        return best_rows

    # Main search loop
    best_rows = None
    best_score = float('inf')
    
    # Try multiple seeds
    # The DFS construction is deterministic for a given seed? 
    # random.seed(seed) is used in construct_rows_greedy_dfs, but the DFS itself is deterministic.
    # The randomness comes from... actually, my DFS is deterministic (tries x in order).
    # So different seeds will produce the same result if the logic is deterministic.
    # To get diversity, we should add randomness to the DFS or the initial greedy.
    
    # Let's modify find_min_row to accept a seed and use it to shuffle the order of x?
    # Or just rely on the local search to break symmetry.
    # For now, let's run the construction once, then local search.
    # If we want diversity, we can perturb the initial solution.
    
    # Let's run construction with a few different "strategies" by varying max_limit or randomizing greedy.
    
    # Strategy 1: Standard
    rows = construct_rows_greedy_dfs(seed=1)
    if rows:
        rows = local_search(rows, max_time=20.0)
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
            
    # Strategy 2: Perturb and Search
    if best_rows:
        # Randomly perturb one element
        perturbed = [list(r) for r in best_rows]
        random.seed(42)
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, len(perturbed[r_idx]) - 1)
        # Try to change it to a nearby value
        old_val = perturbed[r_idx][i_idx]
        for delta in range(1, 10):
            new_val = old_val + delta
            if new_val < perturbed[r_idx][i_idx+1] if i_idx < len(perturbed[r_idx])-1 else True:
                if new_val > perturbed[r_idx][i_idx-1]:
                    # Check validity
                    # This is complex. Let's just use local search on the perturbed version.
                    # But we need a valid perturbation.
                    # Let's just run local search again with a different seed for the search?
                    # The local search is deterministic too.
                    # Let's add a random restart in local search?
                    pass
                    
    # If we have time, run local search again on the best found
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining > 10 and best_rows:
        # Run local search again, maybe with a longer time?
        # Or just return best_rows.
        # The local search already runs for 20s.
        # Let's try to do one more pass if we are close to the bound.
        if best_score <= 135:
            final_rows = local_search(best_rows, max_time=min(remaining - 5, 30.0))
            final_score = get_score(final_rows)
            if final_score < best_score:
                best_score = final_score
                best_rows = [list(r) for r in final_rows]
                
    if best_rows is None:
        # Fallback
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if used & bit:
                        valid = False
                        break
                    new |= bit
                if valid:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    return best_rows
