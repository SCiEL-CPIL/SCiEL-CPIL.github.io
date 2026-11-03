<<<<<<< SEARCH
def _greedy_full(n_rows, k, rng, max_val=None):
    """Build all rows greedily with randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_val)
        if len(row) < k + 1:
            return None
        rows.append(row)
    return rows

def _sa_optimize(rows, k, rng, time_budget=60):
    """Simulated Annealing to minimize the scope (max entry)."""
    n_rows = len(rows)
    # Initial state
    best_rows = [list(r) for r in rows]
    current_rows = [list(r) for r in best_rows]
    
    # Track used differences globally for fast conflict checking
    # We need to be able to add/remove row diffs quickly.
    # Since k is small (5), each row has 15 diffs.
    # Total diffs is manageable.
    
    def get_diffs(row):
        return _row_diffs(row)

    # Initialize global diff counts
    diff_counts = {}
    for row in current_rows:
        for d in get_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
    def count_conflicts():
        c = 0
        for cnt in diff_counts.values():
            if cnt > 1:
                c += cnt - 1
        return c

    def get_scope():
        return _max_entry(current_rows)

    # Objective: Minimize scope + Penalty * Conflicts
    # We want valid solutions (0 conflicts) with low scope.
    # If invalid, we prioritize reducing conflicts first.
    
    current_scope = get_scope()
    current_conflicts = count_conflicts()
    current_cost = current_scope + 1000 * current_conflicts
    
    best_scope = current_scope
    best_conflicts = current_conflicts
    best_cost = current_cost
    
    start_time = time.time()
    temp = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
            
        # Pick a random move
        move_type = rng.random()
        
        if move_type < 0.7:
            # Move: Rebuild a random row
            ri = rng.randrange(n_rows)
            old_row = current_rows[ri]
            
            # Remove old row diffs from counts
            for d in get_diffs(old_row):
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Rebuild row with some randomness, trying to stay within current scope
            # Use a greedy approach with a slight bias towards smaller numbers
            used = set(diff_counts.keys())
            max_limit = current_scope + rng.randint(0, 5) # Allow slight expansion to escape
            new_row = _greedy_row(used, k, rng, max_limit)
            
            if len(new_row) < k + 1:
                # Restore old row
                for d in get_diffs(old_row):
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                continue
            
            # Add new row diffs
            for d in get_diffs(new_row):
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            current_rows[ri] = new_row
            
        else:
            # Move: Shift an entry in a random row
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            if len(row) < 2:
                continue
            
            # Pick a random index > 0
            idx = rng.randint(1, len(row) - 1)
            old_val = row[idx]
            
            # Try to reduce value to compress scope
            # Or randomly perturb
            if rng.random() < 0.6:
                # Try to reduce
                reduction = rng.randint(1, 3)
                new_val = old_val - reduction
                if new_val <= 0:
                    continue
                # Check strictly increasing
                if idx > 1 and new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
            else:
                # Random small change
                delta = rng.randint(-3, 3)
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if idx > 1 and new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
            
            # If new_val == old_val, skip
            if new_val == old_val:
                continue
                
            # Calculate diff changes
            old_diffs = get_diffs(row)
            
            # Remove old diffs
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Apply change
            row[idx] = new_val
            new_row = row
            new_diffs = get_diffs(new_row)
            
            # Add new diffs
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            # Check validity of the row itself (strictly increasing is already checked)
            # But we need to ensure no internal duplicates in the new row?
            # The problem says "increasing integers", so strictly increasing is required.
            # Our check `new_val <= row[idx-1]` etc ensures that.
            
            # Evaluate cost
            new_scope = get_scope()
            new_conflicts = count_conflicts()
            new_cost = new_scope + 1000 * new_conflicts
            
            delta_cost = new_cost - current_cost
            
            # Acceptance criterion
            if delta_cost < 0 or (temp > 0 and rng.random() < pow(2.718281828, -delta_cost / temp)):
                current_scope = new_scope
                current_conflicts = new_conflicts
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = current_scope
                    best_conflicts = current_conflicts
                    best_rows = [list(r) for r in current_rows]
            else:
                # Reject: Revert changes
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    
                row[idx] = old_val
                # current_rows[ri] is already updated to new_val, so revert it
                current_rows[ri][idx] = old_val

        # Cool down
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    # Return the best valid solution found, or the best overall if no valid found
    # The task requires a valid solution.
    # If best_conflicts > 0, we should try to return a valid one from history if possible,
    # or just return the best_rows and hope it's valid? No, it must be valid.
    # Let's check if best_rows is valid.
    if _is_valid(best_rows, k):
        return best_rows
    else:
        # Fallback: try to find any valid solution from the process?
        # Or just return current if valid?
        if _is_valid(current_rows, k):
            return current_rows
        # If neither is valid, return best_rows (which might be invalid, but we hope the greedy init was valid)
        return best_rows
=======
def _greedy_full(n_rows, k, rng, max_val=None):
    """Build all rows greedily with randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_val)
        if len(row) < k + 1:
            return None
        rows.append(row)
    return rows

def _reconstruct_row_from_diffs(diffs, k, rng):
    """Reconstruct a strictly increasing row from a multiset of differences.
    
    Args:
        diffs: list of positive differences (should have exactly k*(k+1)//2 elements)
        k: number of intervals (row length is k+1)
        rng: random number generator for tie-breaking
    
    Returns:
        The row as a list of k+1 integers starting at 0, or None if reconstruction fails.
    """
    expected_count = k * (k + 1) // 2
    if len(diffs) != expected_count:
        return None
    
    # Count occurrences of each difference
    diff_counts = {}
    for d in diffs:
        diff_counts[d] = diff_counts.get(d, 0) + 1
    
    # The largest difference in the row is always the max entry (last element).
    # We'll try to reconstruct the row by choosing each element one by one.
    
    def backtrack(pos, current_row, remaining_counts):
        """Try to build the row from position pos onwards."""
        if pos == k + 1:
            # Check if all differences are used
            for cnt in remaining_counts.values():
                if cnt != 0:
                    return None
            return current_row[:]
        
        if pos == 0:
            # First element is 0
            return backtrack(1, [0], remaining_counts)
        
        # Try each possible value for current_row[pos]
        # The value must be greater than current_row[pos-1]
        # And the differences to all previous elements must be in remaining_counts
        
        # To limit the search, we can try values that are differences from previous elements
        # Specifically, current_row[pos] - current_row[i] for i < pos must be in remaining_counts
        
        # Let's try each difference in remaining_counts as a candidate for current_row[pos] - current_row[pos-1]
        candidates = list(remaining_counts.keys())
        rng.shuffle(candidates)  # Randomize for diversity
        
        for d in candidates:
            if remaining_counts[d] <= 0:
                continue
            candidate = current_row[pos-1] + d
            # Check if all differences to previous elements are in remaining_counts
            valid = True
            needed_diffs = []
            for i in range(pos):
                diff = candidate - current_row[i]
                if diff not in remaining_counts or remaining_counts[diff] <= 0:
                    valid = False
                    break
                needed_diffs.append(diff)
            
            if not valid:
                continue
            
            # Check if we have enough copies of each needed difference
            temp_remaining = dict(remaining_counts)
            for diff in needed_diffs:
                temp_remaining[diff] -= 1
                if temp_remaining[diff] < 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Recurse
            result = backtrack(pos + 1, current_row + [candidate], temp_remaining)
            if result is not None:
                return result
        
        return None
    
    result = backtrack(0, [], diff_counts)
    return result

def _ga_optimize(rows, k, rng, time_budget=60):
    """Genetic Algorithm to minimize the scope (max entry) over difference multisets."""
    n_rows = len(rows)
    
    # Convert initial rows to difference multisets
    def rows_to_diffs(rows):
        all_diffs = []
        for row in rows:
            all_diffs.append(_row_diffs(row))
        return all_diffs
    
    def diffs_to_rows(all_diffs):
        rows = []
        for diffs in all_diffs:
            row = _reconstruct_row_from_diffs(diffs, k, rng)
            if row is None:
                return None
            rows.append(row)
        return rows
    
    # Initialize population
    pop_size = 20
    population = []
    for i in range(pop_size):
        if i == 0:
            # First individual is the initial solution
            population.append(rows_to_diffs(rows))
        else:
            # Mutate the initial solution
            base = rows_to_diffs(rows)
            mutated = []
            for row_diffs in base:
                new_diffs = list(row_diffs)
                # Mutate a few differences
                for _ in range(rng.randint(1, 3)):
                    idx = rng.randrange(len(new_diffs))
                    delta = rng.randint(-5, 5)
                    new_diffs[idx] = max(1, new_diffs[idx] + delta)
                mutated.append(new_diffs)
            population.append(mutated)
    
    # Track best solution
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Evaluate fitness
        def fitness(all_diffs):
            rows = diffs_to_rows(all_diffs)
            if rows is None:
                return float('inf')
            if not _is_valid(rows, k):
                # Penalize invalid solutions heavily
                conflicts = 0
                seen = set()
                for row in rows:
                    for d in _row_diffs(row):
                        if d in seen:
                            conflicts += 1
                        seen.add(d)
                return _max_entry(rows) + 1000 * conflicts
            return _max_entry(rows)
        
        # Sort population by fitness
        scored_pop = []
        for ind in population:
            f = fitness(ind)
            scored_pop.append((f, ind))
        scored_pop.sort(key=lambda x: x[0])
        
        # Update best
        if scored_pop[0][0] < best_scope:
            best_scope = scored_pop[0][0]
            best_rows = diffs_to_rows(scored_pop[0][1])
        
        # Create next generation
        new_population = []
        
        # Elitism: keep the best few
        for i in range(min(3, pop_size)):
            new_population.append(scored_pop[i][1])
        
        # Crossover and mutation to fill the rest
        while len(new_population) < pop_size:
            # Select parents using tournament selection
            parent1 = rng.choice(scored_pop[:pop_size//2])[1]
            parent2 = rng.choice(scored_pop[:pop_size//2])[1]
            
            # Crossover: combine differences from both parents
            child = []
            for i in range(n_rows):
                # For each row, pick differences from either parent
                if rng.random() < 0.5:
                    child.append(list(parent1[i]))
                else:
                    child.append(list(parent2[i]))
            
            # Mutation
            for i in range(n_rows):
                for _ in range(rng.randint(1, 3)):
                    idx = rng.randrange(len(child[i]))
                    delta = rng.randint(-5, 5)
                    child[i][idx] = max(1, child[i][idx] + delta)
            
            new_population.append(child)
        
        population = new_population
    
    # Return the best valid solution
    if best_rows and _is_valid(best_rows, k):
        return best_rows
    else:
        return rows
>>>>>>> REPLACE
