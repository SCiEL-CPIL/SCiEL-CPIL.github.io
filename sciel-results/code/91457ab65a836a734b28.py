# EVOLVE-BLOCK-START
import time
import random

def _interleaved_greedy(n_rows, k, seed=0):
    """Interleaved greedy: build all rows simultaneously, one element at a time."""
    random.seed(seed)
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    # Build up to k+1 elements in each row
    for step in range(1, k + 1):
        # Shuffle row order for diversity
        order = list(range(n_rows))
        random.shuffle(order)
        for r in order:
            row = rows[r]
            x = row[-1] + 1
            while True:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
    
    return rows

def _optimize(rows, n_rows, k, time_limit=100.0, seed=42):
    """
    Optimized local search using difference counts.
    Returns the best rows found.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    # Helper to compute current scope
    def get_scope():
        return max(max(r) for r in current_rows)
    
    # Initialize difference counts
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 20
    diff_counts = [0] * (diff_size + 1)
    
    # Populate initial counts
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    
    def try_move(r, pos, new_val):
        """Attempt to move current_rows[r][pos] to new_val. Returns True if successful."""
        row = current_rows[r]
        old_val = row[pos]
        
        # Check increasing constraint
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False
        
        # Calculate old and new differences
        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == pos:
                continue
            old_diffs.append(abs(old_val - row[i]))
            new_diffs.append(abs(new_val - row[i]))
        
        # Remove old diffs
        for d in old_diffs:
            diff_counts[d] -= 1
            
        # Check new diffs
        valid = True
        for d in new_diffs:
            if d >= len(diff_counts):
                 pass
            if diff_counts[d] > 0:
                valid = False
                break
        
        if not valid:
            # Restore old diffs
            for d in old_diffs:
                diff_counts[d] += 1
            return False
        
        # Apply move
        row[pos] = new_val
        for d in new_diffs:
            diff_counts[d] += 1
            
        return True

    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Iterate rows and positions in random order for diversity
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            # Try to reduce elements from right to left? Or random?
            # Let's try to reduce the largest elements first as they contribute most to scope.
            # But we must maintain order.
            # Let's just iterate all positions.
            for pos in range(len(row) - 1, 0, -1): # Right to left
                if (time.time() - start_time) > time_limit:
                    break
                old_val = row[pos]
                # Try to reduce to smallest valid value
                # Lower bound: row[pos-1] + 1 (if pos > 0)
                lower_bound = row[pos - 1] + 1 if pos > 0 else 1
                # Upper bound: old_val - 1
                # We want the smallest new_val >= lower_bound such that move is valid.
                
                found = False
                # Try values from lower_bound to old_val - 1
                for new_val in range(lower_bound, old_val):
                    if try_move(r, pos, new_val):
                        improved = True
                        found = True
                        break
                if not found:
                    pass
                    
    # Update best
    current_scope = get_scope()
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [list(r) for r in current_rows]
        
    return best_rows

def _optimize_simulated_annealing(rows, n_rows, k, time_limit=100.0, seed=42):
    """
    Simulated annealing local search. Allows worsening moves early on to escape local minima.
    Includes row-replacement moves for larger structural changes.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 20
    diff_counts = [0] * (diff_size + 1)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    current_scope = best_scope
    
    # Annealing parameters
    T = 10.0
    T_min = 0.1
    alpha = 0.995 # Slower cooling to allow more exploration

    def try_row_replacement(r):
        """
        Attempts to replace row r with a new valid row constructed greedily
        using the currently available differences (after removing row r's diffs).
        Returns (success, new_row, old_row, scope_change) or (False, None, None, 0).
        """
        old_row = current_rows[r]
        
        # 1. Remove old row's diffs from counts
        old_diffs = []
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                d = old_row[j] - old_row[i]
                old_diffs.append(d)
                diff_counts[d] -= 1
        
        # 2. Try to construct a new row
        # We want the new row to be as "tight" as possible.
        # Greedy: start with [0], then pick smallest x > last such that diffs are available.
        new_row = [0]
        success = True
        for step in range(1, k + 1):
            x = new_row[-1] + 1
            found = False
            # Limit search space to avoid infinite loops if impossible
            # We can limit x to current_scope + some margin, or just let it run
            # But if we want to minimize scope, we should stop if x > best_scope + margin?
            # Actually, if we are in SA, we might accept a larger scope temporarily.
            # However, if we can't find a row within a reasonable bound, we fail.
            # Let's try up to current_scope + 20. If we can't fit, fail.
            limit = max(current_scope, best_scope) + 20
            
            while x <= limit:
                new_diffs = {x - a for a in new_row}
                # Check if all new diffs are available (count 0)
                valid = True
                for d in new_diffs:
                    if d >= len(diff_counts) or diff_counts[d] > 0:
                        valid = False
                        break
                if valid:
                    new_row.append(x)
                    found = True
                    break
                x += 1
            
            if not found:
                success = False
                break
        
        if not success:
            # Restore old diffs
            for d in old_diffs:
                diff_counts[d] += 1
            return False, None, None, 0
        
        # 3. Calculate new scope
        new_max = max(new_row)
        # The global scope is max(new_max, max of other rows)
        # We don't need to recompute max of all rows if we know the old max
        # But for simplicity and correctness in SA acceptance, let's just return the new row.
        # The caller will update current_rows and check scope.
        
        # For now, we just return the new row and let the caller handle the swap and scope calc.
        # But we need to know if it's better to decide acceptance logic?
        # Standard SA: accept if better, or with prob exp(-delta/T).
        # Delta is change in objective function (scope).
        
        return True, new_row, old_row, 0

    while (time.time() - start_time) < time_limit:
        if T <= T_min:
            break
            
        # Choose move type: 50% single element, 50% row replacement
        if random.random() < 0.5:
            # Single element move
            r = random.randint(0, n_rows - 1)
            pos = random.randint(1, k) # 1 to k
            
            row = current_rows[r]
            old_val = row[pos]
            
            # Generate a random new value within a reasonable range
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            
            lower_bound = row[pos - 1] + 1
            upper_bound = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 20
            
            if new_val < lower_bound:
                new_val = lower_bound
            if new_val > upper_bound:
                new_val = upper_bound
                
            if new_val == old_val:
                continue
                
            old_diffs = []
            new_diffs = []
            for i in range(len(row)):
                if i == pos:
                    continue
                old_diffs.append(abs(old_val - row[i]))
                new_diffs.append(abs(new_val - row[i]))
                
            # Check if move is valid
            for d in old_diffs:
                diff_counts[d] -= 1
                
            valid = True
            for d in new_diffs:
                if d >= len(diff_counts) or diff_counts[d] > 0:
                    valid = False
                    break
            
            if not valid:
                for d in old_diffs:
                    diff_counts[d] += 1
                continue
                
            # Apply move
            row[pos] = new_val
            for d in new_diffs:
                diff_counts[d] += 1
                
            new_scope = get_scope()
            
            if new_scope < current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                delta_e = new_scope - current_scope
                if random.random() < pow(2.71828, -delta_e / T):
                    current_scope = new_scope
                else:
                    # Reject, revert
                    row[pos] = old_val
                    for d in new_diffs:
                        diff_counts[d] -= 1
                    for d in old_diffs:
                        diff_counts[d] += 1
        else:
            # Row replacement move
            r = random.randint(0, n_rows - 1)
            success, new_row, old_row, _ = try_row_replacement(r)
            
            if success:
                # Calculate new scope
                # Temporarily set the row to new_row to get scope?
                # Or just compute max of new_row and other rows.
                # For efficiency, we can just update the row and call get_scope()
                # But we need to handle rejection.
                
                # Store old row to restore if rejected
                # The function already removed old diffs.
                # We need to add new diffs if accepted, or restore old diffs if rejected.
                
                # Let's compute new_scope by temporarily replacing
                # This is O(n_rows * k) which is small.
                
                # Actually, try_row_replacement already removed old diffs.
                # It did NOT add new diffs yet.
                
                # We need to check if new_row is valid with respect to other rows.
                # The greedy construction ensured new_row's internal diffs are available 
                # in the state where old_row was removed. 
                # So new_row is valid with respect to other rows.
                
                # Now, calculate new scope.
                # We can just replace the row in current_rows and call get_scope()
                # But if we reject, we must revert.
                
                # Let's add new diffs to counts temporarily? No, we can just compute scope.
                # Scope is max(max(row) for row in current_rows).
                # If we replace current_rows[r] with new_row, we can compute scope.
                
                # Let's do it carefully.
                # 1. Compute new_scope.
                #    max_other = max(max(row) for i, row in enumerate(current_rows) if i != r)
                #    new_scope = max(max_other, max(new_row))
                
                max_other = 0
                for i in range(n_rows):
                    if i != r:
                        if max(current_rows[i]) > max_other:
                            max_other = max(current_rows[i])
                
                new_scope = max(max_other, max(new_row))
                
                # 2. Acceptance decision
                if new_scope < current_scope:
                    # Accept
                    current_rows[r] = new_row
                    # Add new diffs to counts
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            diff_counts[d] += 1
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                else:
                    delta_e = new_scope - current_scope
                    if random.random() < pow(2.71828, -delta_e / T):
                        # Accept
                        current_rows[r] = new_row
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                d = new_row[j] - new_row[i]
                                diff_counts[d] += 1
                        current_scope = new_scope
                    else:
                        # Reject
                        # Restore old diffs
                        for d in old_row:
                            pass # We need the diffs of old_row
                        # We stored old_diffs in try_row_replacement? No, we didn't return them.
                        # We need to recompute or store them.
                        
                        # Let's recompute old_diffs from old_row (which is still available)
                        for i in range(len(old_row)):
                            for j in range(i + 1, len(old_row)):
                                d = old_row[j] - old_row[i]
                                diff_counts[d] += 1
                        # current_rows[r] is still old_row, so no change needed there.
        
        T *= alpha
        
    return best_rows

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_rows = None
    best_scope = float('inf')
    
    # 1. Generate diverse initial solutions using interleaved greedy
    max_greedy_seeds = 10
    
    for seed in range(max_greedy_seeds):
        if time.time() > deadline - 60.0: # Leave 60s for optimization
            break
            
        rows = _interleaved_greedy(n_rows, k, seed=seed)
        sc = max(max(r) for r in rows)
        
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in rows]
            
    if best_rows is None:
        # Fallback to simple greedy
        rows = []
        used = set()
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # 2. Optimization Phase
    time_remaining = deadline - time.time()
    
    if time_remaining > 10.0:
        # Run Hill Climbing
        t_hc = time_remaining * 0.3
        if t_hc > 5.0:
            optimized_hc = _optimize(best_rows, n_rows, k, time_limit=t_hc, seed=42)
            sc_hc = max(max(r) for r in optimized_hc)
            if sc_hc < best_scope:
                best_scope = sc_hc
                best_rows = optimized_hc
        
        # Run SA on the current best
        t_sa = deadline - time.time() - 5.0 # Leave 5s buffer
        if t_sa > 5.0:
            optimized_sa = _optimize_simulated_annealing(best_rows, n_rows, k, time_limit=t_sa, seed=43)
            sc_sa = max(max(r) for r in optimized_sa)
            if sc_sa < best_scope:
                best_scope = sc_sa
                best_rows = optimized_sa
                
    return best_rows
# EVOLVE-BLOCK-END
