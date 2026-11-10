<<<<<<< SEARCH
    def simulated_annealing(rows, n_rows, k, max_time=10.0):
        """
        Apply Simulated Annealing to minimize the scope of the DTS.
        """
        # Deep copy
        current_rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(current_rows)
        current_scope = max(row[-1] for row in current_rows)
        
        best_rows = [row[:] for row in current_rows]
        best_scope = current_scope
        
        start_time_sa = time.time()
        T = 10.0  # Initial temperature
        cooling_rate = 0.995
        min_T = 0.01
        
        rng = random.Random(int(time.time() * 1000) % 10000)
        
        # Pre-calculate total number of elements
        total_elements = n_rows * (k + 1)
        
        while T > min_T and (time.time() - start_time_sa) < max_time:
            if time.time() > deadline - 5:
                break
                
            # Pick a random row and a random position (not the first one)
            r_idx = rng.randint(0, n_rows - 1)
            c_idx = rng.randint(1, k) # 1 to k inclusive
            
            row = current_rows[r_idx]
            old_val = row[c_idx]
            
            # Determine valid range for new_val
            # Must be > row[c_idx-1] and < row[c_idx+1] (if exists)
            min_val = row[c_idx - 1] + 1
            max_val = row[c_idx + 1] - 1 if c_idx < k else current_scope + 50 # Allow expansion
            
            # If range is too large, restrict it to nearby values to keep search local
            # But we also want to allow "jumps" sometimes
            if max_val - min_val > 20:
                # Randomly pick a window
                window_size = rng.randint(5, 20)
                center = rng.randint(min_val, max_val)
                new_min = max(min_val, center - window_size)
                new_max = min(max_val, center + window_size)
            else:
                new_min = min_val
                new_max = max_val
                
            if new_min > new_max:
                T *= cooling_rate
                continue
                
            new_val = rng.randint(new_min, new_max)
            
            if new_val == old_val:
                T *= cooling_rate
                continue
                
            # Check validity
            # Remove old diffs
            old_diffs = set()
            for i in range(c_idx):
                old_diffs.add(old_val - row[i])
            for j in range(c_idx + 1, k + 1):
                old_diffs.add(row[j] - old_val)
                
            temp_used = used_diffs - old_diffs
            
            # Check new diffs
            new_diffs = set()
            valid = True
            for i in range(c_idx):
                d = new_val - row[i]
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                for j in range(c_idx + 1, k + 1):
                    d = row[j] - new_val
                    if d <= 0 or d in temp_used:
                        valid = False
                        break
                    new_diffs.add(d)
                    
            if not valid:
                T *= cooling_rate
                continue
                
            # Calculate new scope
            # Only the last element of a row affects scope if it's the max
            # But changing an internal element doesn't change the last element directly,
            # unless we are changing the last element.
            # Wait, scope is max of all entries.
            # If we change row[r_idx][c_idx], the scope might change if:
            # 1. c_idx == k (last element)
            # 2. The old value was the unique maximum in the whole set
            # 3. The new value is larger than the previous max
            
            # To be safe and simple, recalculate scope only if necessary or just track max.
            # For k=8, n=4, recalculation is cheap.
            # But we can optimize:
            # If c_idx < k, the last element of this row is unchanged.
            # The global max might change if old_val was the global max.
            
            new_scope = current_scope
            if c_idx == k:
                # Last element changed
                # If new_val < old_val, scope might decrease
                # If new_val > old_val, scope increases
                if new_val > old_val:
                    new_scope = max(current_scope, new_val)
                else:
                    # Need to find new max
                    # If old_val was the unique max, we need to find the next max
                    # This is expensive. Let's just calculate max of all rows.
                    # Optimization: keep a list of last elements? No, any element can be max.
                    # For now, calculate max of all elements in the modified row and compare with others?
                    # Actually, scope is max(all entries).
                    # Let's just compute max of current_rows with the modification.
                    # It's O(n*k) which is 4*9=36 ops. Very fast.
                    max_in_row = max(current_rows[r_idx][:c_idx] + [new_val] + current_rows[r_idx][c_idx+1:])
                    max_others = max(max(row) for i, row in enumerate(current_rows) if i != r_idx)
                    new_scope = max(max_in_row, max_others)
            else:
                # Internal element changed
                # If new_val > current_scope, new_scope = new_val
                if new_val > current_scope:
                    new_scope = new_val
                else:
                    # If old_val was the unique global max, scope might decrease.
                    # Check if old_val == current_scope and it's the only one?
                    # Simple check:
                    if old_val == current_scope:
                         # Count how many times current_scope appears
                         count_max = 0
                         for i, row in enumerate(current_rows):
                             if i == r_idx:
                                # Check modified row
                                if new_val == current_scope: count_max += 1
                                # Other elements in this row?
                                for j in range(k+1):
                                    if j != c_idx and row[j] == current_scope:
                                        count_max += 1
                             else:
                                if current_scope in row:
                                    count_max += row.count(current_scope)
                                    # Actually just check if any other row has it
                                    # For simplicity, just recalc if old_val == current_scope
                                    pass
                         # If count_max > 1, scope stays. If 1, scope decreases.
                         # To be safe and fast, just recalc if old_val == current_scope
                         if count_max <= 1:
                             max_in_row = max(current_rows[r_idx][:c_idx] + [new_val] + current_rows[r_idx][c_idx+1:])
                             max_others = max(max(row) for i, row in enumerate(current_rows) if i != r_idx)
                             new_scope = max(max_in_row, max_others)
                    # Else scope stays same
                    else:
                         new_scope = current_scope

            # Acceptance
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # Accept
                current_rows[r_idx][c_idx] = new_val
                used_diffs = temp_used | new_diffs
                current_scope = new_scope
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [row[:] for row in current_rows]
                    
            T *= cooling_rate
            
        return best_rows, best_scope
=======
    def local_search_reduce_max(rows, n_rows, k, max_time=10.0):
        """
        Apply local search to minimize the scope of the DTS.
        Focuses on reducing the maximum entry by moving elements to smaller valid positions.
        """
        current_rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(current_rows)
        current_scope = max(max(row) for row in current_rows)
        
        best_rows = [row[:] for row in current_rows]
        best_scope = current_scope
        
        start_time_ls = time.time()
        rng = random.Random(int(time.time() * 1000) % 10000)
        
        # We try to improve by finding any move that reduces the scope.
        # If no scope-reducing move is found, we might accept a scope-increasing move
        # with a small probability to escape local minima, but primarily we want to reduce.
        
        while (time.time() - start_time_ls) < max_time:
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try to find a move that reduces the scope
            # We prioritize the largest elements
            # Get list of (value, row_idx, col_idx) sorted by value descending
            elements = []
            for r in range(n_rows):
                for c in range(1, k+1):
                    elements.append((current_rows[r][c], r, c))
            elements.sort(reverse=True)
            
            # Try top few elements
            for val, r_idx, c_idx in elements[:20]:
                if val < current_scope:
                    break # If even the top elements are below current scope, we might be stuck or need a different strategy
                    # Actually, if val < current_scope, moving it won't reduce the global max unless it IS the max.
                    # So we only care about elements that are part of the max or can lead to a reduction.
                    
            # Let's just pick a random element from the top 50% of values to allow exploration
            # But bias towards larger values
            candidates = [e for e in elements if e[0] >= current_scope * 0.8]
            if not candidates:
                candidates = elements[:10]
                
            val, r_idx, c_idx = rng.choice(candidates)
            
            row = current_rows[r_idx]
            old_val = row[c_idx]
            
            # We want to move this element to a smaller value if possible
            # Lower bound is row[c_idx-1] + 1
            # Upper bound is old_val - 1 (to ensure reduction)
            # If old_val is not the unique max, reducing it might not reduce scope, but it's still a valid move.
            # However, if we just want to reduce scope, we should only accept moves that reduce scope.
            # Let's try to find a valid position < old_val.
            
            min_val = row[c_idx - 1] + 1
            max_val = old_val - 1
            
            if min_val > max_val:
                continue
                
            # Try a few random values in [min_val, max_val]
            attempts = 0
            while attempts < 10:
                new_val = rng.randint(min_val, max_val)
                
                # Check validity
                old_diffs = set()
                for i in range(c_idx):
                    old_diffs.add(old_val - row[i])
                for j in range(c_idx + 1, k + 1):
                    old_diffs.add(row[j] - old_val)
                
                temp_used = used_diffs - old_diffs
                
                new_diffs = set()
                valid = True
                for i in range(c_idx):
                    d = new_val - row[i]
                    if d <= 0 or d in temp_used:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    for j in range(c_idx + 1, k + 1):
                        d = row[j] - new_val
                        if d <= 0 or d in temp_used:
                            valid = False
                            break
                        new_diffs.add(d)
                        
                if not valid:
                    attempts += 1
                    continue
                
                # If valid, calculate new scope
                # Since new_val < old_val, the scope can only decrease or stay the same.
                # It decreases if old_val was the unique global maximum.
                # Let's just compute the new scope to be safe.
                # Optimization: If old_val < current_scope, new_scope = current_scope.
                # If old_val == current_scope, we need to check if it's the unique max.
                
                if old_val < current_scope:
                    new_scope = current_scope
                else:
                    # Recalculate max
                    max_in_row = max(current_rows[r_idx][:c_idx] + [new_val] + current_rows[r_idx][c_idx+1:])
                    max_others = max(max(row) for i, row in enumerate(current_rows) if i != r_idx)
                    new_scope = max(max_in_row, max_others)
                
                # Accept if scope improves
                if new_scope < current_scope:
                    current_rows[r_idx][c_idx] = new_val
                    used_diffs = temp_used | new_diffs
                    current_scope = new_scope
                    improved = True
                    
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [row[:] for row in current_rows]
                    break # Restart the element selection loop with updated state
                    
                else:
                    # If scope didn't improve, we could still accept with small probability to escape,
                    # but since we are targeting "reduce max", let's be greedy first.
                    # If we get stuck, we might need a random walk.
                    # For now, just try another value.
                    attempts += 1
                    if attempts >= 10:
                        break
            
            if not improved:
                # If we couldn't find a reduction, maybe do a random move to escape?
                # Or just break if we are stuck for a while?
                # Let's add a simple random walk if stuck
                if rng.random() < 0.1:
                    # Pick a random element and move it randomly (valid move)
                    r_idx = rng.randint(0, n_rows - 1)
                    c_idx = rng.randint(1, k)
                    row = current_rows[r_idx]
                    old_val = row[c_idx]
                    min_val = row[c_idx - 1] + 1
                    max_val = row[c_idx + 1] - 1 if c_idx < k else current_scope + 10
                    if min_val <= max_val:
                        new_val = rng.randint(min_val, max_val)
                        if new_val != old_val:
                            # Check validity
                            old_diffs = set()
                            for i in range(c_idx):
                                old_diffs.add(old_val - row[i])
                            for j in range(c_idx + 1, k + 1):
                                old_diffs.add(row[j] - old_val)
                            temp_used = used_diffs - old_diffs
                            new_diffs = set()
                            valid = True
                            for i in range(c_idx):
                                d = new_val - row[i]
                                if d <= 0 or d in temp_used:
                                    valid = False
                                    break
                                new_diffs.add(d)
                            if valid:
                                for j in range(c_idx + 1, k + 1):
                                    d = row[j] - new_val
                                    if d <= 0 or d in temp_used:
                                        valid = False
                                        break
                                    new_diffs.add(d)
                            if valid:
                                current_rows[r_idx][c_idx] = new_val
                                used_diffs = temp_used | new_diffs
                                current_scope = max(max(row) for row in current_rows)
                                if current_scope < best_scope:
                                    best_scope = current_scope
                                    best_rows = [row[:] for row in current_rows]

        return best_rows, best_scope
>>>>>>> REPLACE

