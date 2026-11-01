import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 138.0  # Keep a margin from the 150s limit

    # Total differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def greedy_init(rand_prob=0.15, window=30):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 300
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    # Phase 1: Get a baseline
    best_rows = greedy_init(0.15, 30)
    best_scope = get_scope(best_rows)

    # Backtracking search with time limit
    def backtracking_search(target_scope, time_limit, seed=0):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses randomized backtracking.
        Returns rows if found, None otherwise.
        """
        random.seed(seed)
        start = time.time()
        
        # State
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        used_diffs = set()
        
        # We will build row by row.
        # To allow randomization in value selection, we'll collect a few valid candidates.
        
        def solve(row_idx, pos):
            if time.time() > start + time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == k + 1:
                return solve(row_idx + 1, 1)
            
            prev_val = rows[row_idx][pos - 1]
            # Lower bound is prev_val + 1
            # Upper bound is target_scope
            lower = prev_val + 1
            upper = target_scope
            
            if lower > upper:
                return False
            
            # Collect valid candidates
            # We bias towards smaller values to keep scope low, 
            # but we take a random sample to escape local minima.
            # Strategy: Find the first 5-10 valid values, pick one randomly.
            # If no valid values in [lower, lower+50], expand search.
            
            candidates = []
            # Scan a window first
            scan_limit = min(upper, lower + 50)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 10:
                        break
            
            if not candidates:
                # If none found in the small window, scan the rest
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 10:
                            break
            
            if not candidates:
                return False
            
            # Shuffle candidates to randomize the search path
            # But we want to bias towards smaller values.
            # We can sort and then pick a random one from the first half, or just shuffle.
            # Shuffling is simpler and explores more.
            random.shuffle(candidates)
            
            for val in candidates:
                # Check validity again (in case we picked a bad one? No, we pre-validated)
                # But we need to check against ALL previous elements in the row, not just the immediate previous.
                # The pre-validation loop above did `for i in range(pos)`, so it is correct.
                
                # Add differences
                new_diffs = []
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    new_diffs.append(d)
                    used_diffs.add(d)
                
                rows[row_idx][pos] = val
                
                if solve(row_idx, pos + 1):
                    return True
                
                # Backtrack
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    # Phase 2: Iterative Improvement using Backtracking
    # We try to reduce the scope by 1 each time.
    # If backtracking fails, we might try a "jolt" to the current best and retry,
    # but since backtracking is global (builds from scratch), it doesn't depend on previous rows' specific values, 
    # only on the target scope. 
    # Wait, the backtracking search builds ALL rows from scratch. 
    # This is good because it finds a globally consistent set for the target scope.
    # It doesn't get stuck in "bad row 0" local minima because it backtracks row 0 too.
    
    # However, pure backtracking might be slow if the target scope is tight.
    # We will allocate time to each attempt.
    
    current_target = best_scope - 1
    seed_counter = 0
    
    while time.time() < deadline - 5:
        if current_target < 112: # Theoretical bound
            break
            
        # Allocate time for this search attempt
        # Start with 2s, increase if we are close to the best?
        # Let's use a fixed chunk of time, e.g., 5s, or adaptive.
        # If we find a solution, we reduce target by 1 and continue.
        # If we fail, we might try a slightly higher target? No, we want to minimize.
        
        # Adaptive time limit: give more time if we are close to the theoretical bound
        time_chunk = 5.0
        if current_target < best_scope - 10:
            time_chunk = 10.0
        elif current_target < best_scope - 20:
            time_chunk = 15.0
        
        # Ensure we don't exceed the deadline
        if time.time() + time_chunk > deadline - 5:
            time_chunk = max(1.0, deadline - 5 - time.time())
        
        result = backtracking_search(current_target, time_chunk, seed=seed_counter)
        seed_counter += 1
        
        if result:
            actual_scope = get_scope(result)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = result
                # If we found a solution, try to reduce target further
                current_target = actual_scope - 1
            else:
                # This shouldn't happen if the search is correct, but just in case
                current_target -= 1
        else:
            # Failed to find a solution with this target
            # Try a slightly higher target to see if we can improve the current best
            if current_target < best_scope:
                current_target += 1
                # But don't go above the current best
                if current_target >= best_scope:
                    break
            else:
                # We are stuck, try a different seed or break
                if seed_counter % 5 == 0:
                    # Try a different random seed
                    pass
                else:
                    # Give up on this target
                    current_target -= 1
                    if current_target < 112:
                        break

    # Phase 3: Final polish with local search if time allows
    if time.time() < deadline - 10:
        # Try to improve the best solution with local search
        def is_valid_move(current_rows, row_idx, pos, new_val):
            """
            Check if changing current_rows[row_idx][pos] to new_val is valid.
            Returns True if the move doesn't create any duplicate differences.
            """
            row = current_rows[row_idx]
            
            # Ensure strictly increasing
            if pos > 0 and new_val <= row[pos - 1]:
                return False
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                return False
            
            # Compute old differences involving position pos
            old_diffs = set()
            for j in range(len(row)):
                if j != pos:
                    if j < pos:
                        old_diffs.add(row[pos] - row[j])
                    else:
                        old_diffs.add(row[j] - row[pos])
            
            # Compute new differences involving position pos
            new_diffs = set()
            for j in range(len(row)):
                if j != pos:
                    if j < pos:
                        new_diffs.add(new_val - row[j])
                    else:
                        new_diffs.add(row[j] - new_val)
            
            # Check for internal conflicts in the same row (differences not involving pos)
            same_row_other_diffs = set()
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    if a != pos and b != pos:
                        same_row_other_diffs.add(row[b] - row[a])
            
            # New diffs must not conflict with same-row other diffs
            if new_diffs & same_row_other_diffs:
                return False
            
            # Check conflicts with other rows
            # Build set of all diffs from other rows
            other_diffs = set()
            for i in range(n_rows):
                if i == row_idx:
                    continue
                r = current_rows[i]
                for a in range(len(r)):
                    for b in range(a + 1, len(r)):
                        other_diffs.add(r[b] - r[a])
            
            # New diffs must not conflict with other rows' diffs
            if new_diffs & other_diffs:
                return False
            
            return True

        def local_search(rows, max_iterations=2000):
            """
            Iterated Local Search: repeatedly try single-element moves to reduce scope.
            Returns the best rows found.
            """
            current_rows = [r[:] for r in rows]
            current_scope = get_scope(current_rows)
            best_rows_ls = [r[:] for r in current_rows]
            best_scope_ls = current_scope
            
            for it in range(max_iterations):
                if time.time() > deadline - 5.0:
                    break
                
                # Try to find a single move that reduces the scope
                found_improvement = False
                
                # Collect candidate moves: focus on rows that achieve the current scope
                scope_rows = [i for i in range(n_rows) if max(current_rows[i]) == current_scope]
                
                # Try moves on scope-achieving rows first, then others
                row_order = scope_rows + [i for i in range(n_rows) if i not in scope_rows]
                random.shuffle(row_order)
                
                for row_idx in row_order:
                    if time.time() > deadline - 5.0:
                        break
                    if found_improvement:
                        break
                    
                    row = current_rows[row_idx]
                    row_max = max(row)
                    
                    # For each position, try to reduce the value
                    # Focus on positions that contribute to the max
                    for pos in range(len(row) - 1, 0, -1):  # Try from end to start
                        if time.time() > deadline - 5.0:
                            break
                        if found_improvement:
                            break
                        
                        old_val = row[pos]
                        
                        # Try reducing this value
                        # How much can we reduce? Must stay > row[pos-1]
                        min_val = row[pos - 1] + 1 if pos > 0 else 1
                        # Also must stay < row[pos+1] if pos < len-1
                        max_allowed = row[pos + 1] - 1 if pos < len(row) - 1 else old_val
                        
                        # Try values from old_val-1 down to min_val
                        # Limit search range for efficiency
                        search_lo = max(min_val, old_val - 15)
                        search_hi = min(max_allowed, old_val - 1)
                        
                        if search_lo > search_hi:
                            continue
                        
                        # Try a few random candidates in range plus the best (smallest) option
                        candidates = set()
                        candidates.add(search_hi)  # smallest valid (most reduction)
                        # Add some random candidates
                        for _ in range(5):
                            c = random.randint(search_lo, search_hi)
                            candidates.add(c)
                        
                        for new_val in sorted(candidates, key=lambda x: x):  # Try most reducing first
                            if is_valid_move(current_rows, row_idx, pos, new_val):
                                # Check if this improves scope
                                new_row_max = max(row[:pos] + [new_val] + row[pos+1:])
                                new_scope = max(
                                    max(max(r) for i, r in enumerate(current_rows) if i != row_idx),
                                    new_row_max
                                )
                                if new_scope < current_scope:
                                    # Accept the move
                                    current_rows[row_idx][pos] = new_val
                                    current_scope = new_scope
                                    if current_scope < best_scope_ls:
                                        best_scope_ls = current_scope
                                        best_rows_ls = [r[:] for r in current_rows]
                                    found_improvement = True
                                    break
                                elif new_scope == current_scope:
                                    # Accept neutral moves sometimes to escape local optima
                                    if random.random() < 0.3:
                                        current_rows[row_idx][pos] = new_val
                                        # current_scope unchanged
                                        break
                            # If new_val is too high, no point trying higher values
                            if new_val > old_val - 1:
                                break
                    # End position loop
                # End row loop
                
                if not found_improvement:
                    # No improvement found; apply a kick perturbation
                    # Randomly perturb 1-3 elements in random rows
                    n_kicks = random.randint(1, 3)
                    for _ in range(n_kicks):
                        if time.time() > deadline - 5.0:
                            break
                        row_idx = random.randint(0, n_rows - 1)
                        row = current_rows[row_idx]
                        if len(row) <= 1:
                            continue
                        pos = random.randint(1, len(row) - 1)
                        old_val = row[pos]
                        
                        # Generate a new random value
                        min_val = row[pos - 1] + 1 if pos > 0 else 1
                        max_val = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 20
                        
                        if min_val > max_val:
                            continue
                        
                        # Try several random values
                        for attempt in range(20):
                            new_val = random.randint(min_val, max_val)
                            if new_val == old_val:
                                continue
                            if is_valid_move(current_rows, row_idx, pos, new_val):
                                current_rows[row_idx][pos] = new_val
                                current_scope = get_scope(current_rows)
                                if current_scope < best_scope_ls:
                                    best_scope_ls = current_scope
                                    best_rows_ls = [r[:] for r in current_rows]
                                break
            # End iteration loop
            
            return best_rows_ls, best_scope_ls

        improved_rows, improved_scope = local_search(best_rows, max_iterations=3000)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    return best_rows
