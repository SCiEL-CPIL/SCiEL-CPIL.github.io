import random
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    time_limit = 110  # Leave margin for 120s limit
    
    # Helper to get diffs of a row
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_row_valid(row):
        diffs = get_diffs(row)
        return len(diffs) == (len(row) * (len(row) - 1)) // 2

    # Improved Initialization: Randomized Greedy with Restart
    def greedy_init():
        best_rows = None
        best_max = float('inf')
        rng = random.Random(42)
        
        for _ in range(50): # Try a few random greedy paths
            used = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                for _ in range(k):
                    # Randomize search order slightly to avoid deterministic traps
                    candidates = list(range(row[-1] + 1, 200))
                    rng.shuffle(candidates)
                    found = False
                    for x in candidates:
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) != len(row):
                            continue
                        if new_diffs & used:
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        success = False
                        break
                if not success:
                    break
                rows.append(row)
            
            if success:
                curr_max = max(max(r) for r in rows)
                if curr_max < best_max:
                    best_max = curr_max
                    best_rows = [list(r) for r in rows]
        
        if best_rows is None:
            # Fallback to simple greedy if randomized fails (unlikely)
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                for _ in range(k):
                    for x in range(row[-1] + 1, 500):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            row.append(x)
                            used.update(new_diffs)
                            break
                rows.append(row)
            best_rows = rows
            best_max = max(max(r) for r in rows)
            
        return best_rows, best_max

    initial_rows, initial_max = greedy_init()
    current_rows = [list(r) for r in initial_rows]
    current_max = initial_max
    best_rows = [list(r) for r in current_rows]
    best_max = current_max
    
    # Calculate initial used diffs
    current_used = set()
    for r in current_rows:
        current_used.update(get_diffs(r))

    # New Mutation Strategy: Row Reconstruction
    # Try to find a valid row with max <= limit that is disjoint from used_diffs
    def find_row_reconstruct(limit, used_diffs, k, rng, prefer_small=True):
        """
        Recursively find a row of length k+1 starting at 0, max <= limit,
        with unique diffs not in used_diffs.
        Returns row or None.
        """
        row = [0]
        used_local = set() # Diffs internal to this new row
        
        def backtrack(idx, current_max_val):
            if idx == k:
                # Check if we have enough elements? 
                # We need k+1 elements total.
                # We started with 0. We need k more.
                # So if idx == k, we have added k elements.
                # Row length is k+1.
                # Diffs are already checked during addition.
                return list(row)
            
            start_val = row[-1] + 1
            end_val = min(limit, current_max_val)
            
            # If prefer_small, try small values first, else random
            candidates = list(range(start_val, end_val + 1))
            if prefer_small:
                pass # Keep order
            else:
                rng.shuffle(candidates)
                
            for x in candidates:
                # Calculate new diffs
                new_diffs = {x - a for a in row}
                
                # Check internal uniqueness
                if len(new_diffs) != len(row):
                    continue
                
                # Check global uniqueness (against other rows)
                if new_diffs & used_diffs:
                    continue
                
                # Check internal uniqueness against existing local diffs?
                # The check `len(new_diffs) != len(row)` ensures new diffs are unique among themselves.
                # But we also need them to be unique against `used_local`?
                # No, `new_diffs` are diffs between x and previous elements.
                # We must ensure these new diffs don't conflict with `used_local` (diffs from previous steps).
                # Wait, `used_local` is not being passed/maintained correctly in this simple sketch.
                # Let's maintain `row_diffs` set.
                
                # Actually, the check `len(new_diffs) == len(row)` ensures x - a_i are all distinct.
                # But do they conflict with diffs generated in previous steps (e.g., a_2 - a_1)?
                # Yes. We need to check `new_diffs & row_diffs` is empty.
                
                # Let's pass row_diffs as a set.
                
                # Revised Backtrack
                return False # Placeholder
            
        # Let's implement a cleaner recursive function
        def _backtrack(row, row_diffs):
            if len(row) == k + 1:
                return list(row)
            
            start_val = row[-1] + 1
            end_val = limit
            
            candidates = list(range(start_val, end_val + 1))
            if not prefer_small:
                rng.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in row}
                
                # 1. Internal uniqueness: new_diffs must be disjoint from row_diffs
                if new_diffs & row_diffs:
                    continue
                # 2. Global uniqueness: new_diffs must be disjoint from used_diffs
                if new_diffs & used_diffs:
                    continue
                # 3. Internal uniqueness of new_diffs themselves
                if len(new_diffs) != len(row):
                    continue
                
                row.append(x)
                row_diffs.update(new_diffs)
                
                res = _backtrack(row, row_diffs)
                if res:
                    return res
                
                row.pop()
                row_diffs.difference_update(new_diffs)
            
            return None
        
        return _backtrack([0], set())

    rng = random.Random(42)
    temperature = 50.0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        if temperature < 0.1:
            break
            
        row_idx = rng.randint(0, n_rows - 1)
        old_row = current_rows[row_idx]
        old_diffs = get_diffs(old_row)
        current_used -= old_diffs
        
        # Determine target max for the new row
        # Try to shrink: limit = current_max - 1
        # If that fails, try current_max
        # If that fails, allow expansion (SA)
        
        new_row = None
        
        # Strategy 1: Try to find a row with max < current_max (Shrink)
        # Only attempt if current_max > some lower bound
        if current_max > 50: # Heuristic lower bound
            # Try a few times to find a smaller max row
            for _ in range(10):
                candidate = find_row_reconstruct(current_max - 1, current_used, k, rng, prefer_small=True)
                if candidate:
                    new_row = candidate
                    break
        
        # Strategy 2: If no shrink found, try to find a row with max == current_max
        if new_row is None:
            # We need to find *any* valid row with max <= current_max
            # This might be hard if the current configuration is tight.
            # We try to find one. If we find one, it's a neutral move (delta=0) or improvement if it frees up space?
            # No, if max stays same, delta=0.
            # But we need to ensure we don't get stuck.
            candidate = find_row_reconstruct(current_max, current_used, k, rng, prefer_small=True)
            if candidate:
                new_row = candidate
        
        # Strategy 3: If still none, allow expansion to escape local optimum
        if new_row is None:
            # Try to find a row with max = current_max + random(1, 10)
            expansion = rng.randint(1, 10)
            target_max = current_max + expansion
            candidate = find_row_reconstruct(target_max, current_used, k, rng, prefer_small=False)
            if candidate:
                new_row = candidate
        
        if new_row is None:
            # Failed to find any row. Restore old.
            current_used |= old_diffs
            temperature *= cooling_rate
            continue
        
        # Calculate new global max
        new_global_max = current_max
        if max(new_row) > new_global_max:
            new_global_max = max(new_row)
        # Note: If the old row was the unique max holder and we replaced it with a smaller max,
        # new_global_max should be the max of the *other* rows.
        # We need to calculate the max of the other rows to be precise.
        max_other = 0
        for i in range(n_rows):
            if i != row_idx:
                if max(current_rows[i]) > max_other:
                    max_other = max(current_rows[i])
        
        actual_new_global_max = max(max_other, max(new_row))
        
        delta = actual_new_global_max - current_max
        
        # Acceptance
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current_rows[row_idx] = new_row
            current_used |= get_diffs(new_row)
            current_max = actual_new_global_max
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in current_rows]
        else:
            current_used |= old_diffs
            
        temperature *= cooling_rate
        
    return best_rows
