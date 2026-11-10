"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    # Known best for (4,7) is 134. We aim to find something <= 134, ideally better.
    # Use backtracking with constraint propagation and randomized restarts.
    
    # First, get a baseline with simple greedy
    best_rows = _greedy_construct(n_rows, k)
    best_max = max(max(row) for row in best_rows)
    
    # Backtracking search with time limit
    # Strategy: build rows incrementally, at each step try smallest valid number first,
    # but with randomized ordering and constraint propagation.
    
    def backtrack_search(max_limit, random_seed=0, time_limit=None):
        """
        Try to find a valid configuration with max entry <= max_limit.
        Uses backtracking with MRV (minimum remaining values) heuristic.
        Returns list of rows if found, None otherwise.
        """
        if time_limit is None:
            time_limit = deadline
        
        random.seed(random_seed)
        
        # State: rows is list of lists, used_diffs is set of all differences used
        # We'll build row by row, but within each row, we pick the next number.
        # To make it more effective, we can interleave rows: at each step, 
        # pick the row with the fewest valid next candidates (MRV).
        
        rows = [[] for _ in range(n_rows)]
        for i in range(n_rows):
            rows[i].append(0)
        
        used_diffs = set()
        
        def get_valid_next(row, last_val, max_allowed):
            """Get all valid next values for a row, up to max_allowed."""
            valid_vals = []
            x = last_val + 1
            while x <= max_allowed:
                # Check if x can be added to row
                ok = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        ok = False
                        break
                if ok:
                    valid_vals.append(x)
                x += 1
            return valid_vals
        
        def count_remaining(row, last_val, max_allowed):
            """Count how many valid next values exist for a row."""
            count = 0
            x = last_val + 1
            while x <= max_allowed:
                ok = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        ok = False
                        break
                if ok:
                    count += 1
                x += 1
            return count
        
        # Backtracking state
        def _backtrack(row_idx, depth):
            # Check time
            if time.time() > time_limit:
                return False
            
            # If all rows are complete
            if depth == n_rows * k:
                return True
            
            # Find which row to extend next using MRV
            # Count remaining positions for each incomplete row
            incomplete_rows = []
            for i in range(n_rows):
                if len(rows[i]) < k + 1:
                    incomplete_rows.append(i)
            
            if not incomplete_rows:
                return True
            
            # Pick row with minimum valid options (MRV)
            min_count = float('inf')
            chosen_row = -1
            chosen_vals = None
            
            for i in incomplete_rows:
                last = rows[i][-1]
                # Count valid values (limit search for speed)
                count = 0
                x = last + 1
                while x <= max_limit and count < 50:  # Cap for speed
                    ok = True
                    for a in rows[i]:
                        d = x - a
                        if d in used_diffs:
                            ok = False
                            break
                    if ok:
                        count += 1
                    x += 1
                
                if count < min_count:
                    min_count = count
                    chosen_row = i
                    if min_count == 0:
                        break
            
            if chosen_row == -1 or min_count == 0:
                return False
            
            # Get valid values for chosen_row
            last = rows[chosen_row][-1]
            valid_vals = []
            x = last + 1
            while x <= max_limit:
                ok = True
                for a in rows[chosen_row]:
                    d = x - a
                    if d in used_diffs:
                        ok = False
                        break
                if ok:
                    valid_vals.append(x)
                x += 1
            
            if not valid_vals:
                return False
            
            # Shuffle for randomization (but keep small values preferred with probability)
            # Use a biased shuffle: small values have higher chance of being tried first
            random.shuffle(valid_vals)
            # But also try the smallest few first for efficiency
            valid_vals.sort()
            # Mix: try smallest, then random, then rest
            if len(valid_vals) > 3:
                # Try first few in order, then shuffle the rest
                head = valid_vals[:3]
                tail = valid_vals[3:]
                random.shuffle(tail)
                valid_vals = head + tail
            
            for val in valid_vals:
                # Add val to chosen_row
                rows[chosen_row].append(val)
                new_diffs = set()
                for a in rows[chosen_row]:
                    if a != val:
                        d = val - a
                        new_diffs.add(d)
                used_diffs.update(new_diffs)
                
                # Recurse
                if _backtrack(chosen_row, depth + 1):
                    return True
                
                # Undo
                rows[chosen_row].pop()
                used_diffs.difference_update(new_diffs)
                
                if time.time() > time_limit:
                    return False
            
            return False
        
        success = _backtrack(0, 0)
        if success and all(len(r) == k + 1 for r in rows):
            return rows
        return None
    
    # Try different max_limit values, starting from best known and going down
    # Also try randomized restarts for each limit
    
    # First, try to beat the current best
    current_best = best_max
    
    # Try limits from current_best-1 down to the counting bound
    # Counting bound for (4,7): C(8,2)*4 = 28*4 = 112, but that's a lower bound on scope
    # Actually the counting argument gives a lower bound on the max value.
    # For n rows of k+1 elements, total distinct diffs = n*C(k+1,2) = 4*28 = 112
    # Max diff is m, so we need at least m >= 112, so m >= 112 is a lower bound? 
    # No, the diffs are all in [1, m], so we need m >= 112.
    
    lower_bound = 112
    
    # Try to find solutions with max <= current_best - 1, then -2, etc.
    # But backtracking can be expensive, so use time limits per attempt.
    
    attempts_per_limit = 20
    time_per_attempt = 3.0  # seconds
    
    for limit in range(current_best - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(attempts_per_limit):
            if time.time() > deadline:
                break
            
            seed = random.randint(0, 2**31 - 1)
            rows = backtrack_search(limit, seed, time_limit=start_time + deadline - start_time)
            
            if rows:
                actual_max = max(max(r) for r in rows)
                if actual_max < best_max:
                    best_max = actual_max
                    best_rows = rows
                found = True
                break
        
        if found:
            # Found a better solution, continue trying to improve
            current_best = best_max
            # Reset attempt counter logic
            # Continue the loop with new current_best
            # But we need to restart the outer loop
            # Let's just continue and the range will be recomputed
            # Actually, the range is fixed, so let's break and restart
            break
    
    # If we found something better, try to improve further
    if best_max < max(max(r) for r in _greedy_construct(n_rows, k)):
        # We improved, try a few more randomized restarts with the current best limit
        for _ in range(10):
            if time.time() > deadline:
                break
            seed = random.randint(0, 2**31 - 1)
            rows = backtrack_search(best_max - 1, seed, time_limit=deadline)
            if rows:
                actual_max = max(max(r) for r in rows)
                if actual_max < best_max:
                    best_max = actual_max
                    best_rows = rows
    
    return best_rows

def _greedy_construct(n_rows, k):
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
    return rows

# EVOLVE-BLOCK-END
