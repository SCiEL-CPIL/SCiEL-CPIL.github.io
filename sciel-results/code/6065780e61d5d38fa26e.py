import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    # 1. Fast Greedy Construction
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_row_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                forbidden = used | current_row_diffs
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        row.append(x)
                        current_row_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
            used |= current_row_diffs
        return rows

    # 2. Backtracking Search with Bitsets
    def backtracking_search(n_rows, k, target_scope, rng, time_limit):
        """
        Attempts to construct n_rows rows of length k+1 within [0, target_scope]
        such that all differences are distinct.
        Returns the list of rows if successful, None otherwise.
        """
        k_plus_1 = k + 1
        # Bitset of used differences. Bit d is 1 if difference d is used.
        # Max difference is target_scope.
        used_bits = 0
        rows = []
        
        # We need to place n_rows * k elements (excluding the 0s)
        # Total differences to place: n_rows * k * (k+1) / 2
        
        def backtrack_row(row_idx, pos, current_row, current_row_bits):
            """
            Recursively fill the current_row.
            row_idx: which row we are filling
            pos: which element in the row we are filling (0 is already 0)
            current_row: list of elements placed so far
            current_row_bits: bitset of differences within current_row
            """
            if time.time() > time_limit:
                return False
            
            if pos == k_plus_1:
                # Row complete
                rows.append(current_row[:])
                return True
            
            # Determine range for next element x
            # x must be > current_row[-1]
            # x must be <= target_scope
            # Also, we need enough room for remaining elements? 
            # Simple bound: x + (k_plus_1 - pos - 1) <= target_scope
            
            start_x = current_row[-1] + 1
            end_x = target_scope - (k_plus_1 - pos - 1)
            
            if start_x > end_x:
                return False
            
            # Generate candidates
            # We want to try smaller x first to keep scope low, but randomize to escape traps
            candidates = list(range(start_x, end_x + 1))
            
            # Heuristic: Shuffle but keep small values more likely? 
            # Or just shuffle fully.
            rng.shuffle(candidates)
            
            # Limit candidates to avoid explosion in early steps
            # If pos is small, we have many choices. If pos is large, fewer.
            # Taking a subset can help speed up but might miss solutions.
            # For now, try all valid candidates in random order, but break early if too many?
            # Actually, checking validity is fast with bitsets.
            
            # To improve performance, we can prioritize candidates that create "rare" differences?
            # For simplicity, we just iterate through shuffled candidates.
            
            # Optimization: If the list is very long, maybe sample?
            # But for k=8, n_rows=4, the space is constrained by target_scope.
            
            for x in candidates:
                # Check if x creates any conflict with global used_bits or current_row_bits
                conflict = False
                new_diffs_bits = 0
                
                for a in current_row:
                    d = x - a
                    if d > target_scope: # Should not happen if end_x is calculated correctly
                        conflict = True
                        break
                    bit = 1 << d
                    # Check if difference d is already used globally
                    if used_bits & bit:
                        conflict = True
                        break
                    # Check if difference d is already in current_row
                    if current_row_bits & bit:
                        conflict = True
                        break
                    new_diffs_bits |= bit
                
                if conflict:
                    continue
                
                # Tentatively accept x
                current_row.append(x)
                new_global_used_bits = used_bits | new_diffs_bits
                new_current_row_bits = current_row_bits | new_diffs_bits
                
                # Recurse
                if backtrack_row(row_idx, pos + 1, current_row, new_current_row_bits):
                    return True
                
                # Backtrack
                current_row.pop()
                
            return False

        # We need to manage 'used_bits' across rows.
        # The above nested function doesn't update 'used_bits' in the outer scope correctly for backtracking across rows easily.
        # Let's refactor to pass used_bits as a parameter or use a nonlocal variable with careful backtracking.
        
        # Refactored Backtracking
        def bt_row_idx(row_idx, pos, current_row, current_row_bits, global_used_bits):
            if time.time() > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == 0:
                # Start new row
                current_row = [0]
                current_row_bits = 0
                return bt_row_idx(row_idx, 1, current_row, current_row_bits, global_used_bits)
            
            if pos == k_plus_1:
                # Row complete
                rows.append(current_row[:])
                # Update global used bits for next row
                return bt_row_idx(row_idx + 1, 0, [], 0, global_used_bits | current_row_bits)
            
            start_x = current_row[-1] + 1
            end_x = target_scope - (k_plus_1 - pos - 1)
            
            if start_x > end_x:
                return False
            
            candidates = list(range(start_x, end_x + 1))
            rng.shuffle(candidates)
            
            # To prevent explosion, if there are too many candidates, we might want to be smarter.
            # But for small k, it's manageable.
            
            for x in candidates:
                conflict = False
                new_diffs_bits = 0
                
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if (global_used_bits & bit) or (current_row_bits & bit):
                        conflict = True
                        break
                    new_diffs_bits |= bit
                
                if conflict:
                    continue
                
                current_row.append(x)
                
                if bt_row_idx(row_idx, pos + 1, current_row, current_row_bits | new_diffs_bits, global_used_bits):
                    return True
                
                current_row.pop()
            
            return False

        rows = []
        success = bt_row_idx(0, 0, [], 0, 0)
        
        if success:
            return rows
        return None

    # 3. Local Compaction
    def local_compact(rows, used_diffs, max_iterations=2000, time_limit=None):
        """Try to reduce the scope by moving elements to smaller positions."""
        if time_limit is None:
            time_limit = time.time() + 5.0
            
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < time_limit:
            improved = False
            iterations += 1
            
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for p_idx in range(len(row) - 1, 0, -1):  # Start from last element
                    old_val = row[p_idx]
                    if old_val <= 1:
                        continue
                    
                    # Remove old_val from row
                    row.pop(p_idx)
                    
                    # Remove its differences from used_diffs
                    diffs_to_remove = set()
                    for a in row:
                        d = abs(old_val - a)
                        diffs_to_remove.add(d)
                    used_diffs -= diffs_to_remove
                    
                    # Try to find a smaller position
                    found = False
                    for x in range(old_val - 1, 0, -1):
                        if x in row:
                            continue
                        valid = True
                        for a in row:
                            d = abs(x - a)
                            if d == 0 or d in used_diffs:
                                valid = False
                                break
                        if valid:
                            # Insert x in sorted position
                            pos = 0
                            while pos < len(row) and row[pos] < x:
                                pos += 1
                            row.insert(pos, x)
                            for a in row:
                                if a != x:
                                    used_diffs.add(abs(x - a))
                            found = True
                            improved = True
                            break
                    
                    if not found:
                        # Re-insert old_val
                        pos = 0
                        while pos < len(row) and row[pos] < old_val:
                            pos += 1
                        row.insert(pos, old_val)
                        for a in row:
                            if a != old_val:
                                used_diffs.add(abs(old_val - a))
        
        return rows, used_diffs

    # 4. SA Polish
    def sa_polish(rows, used_diffs, time_limit):
        """Simulated Annealing to reduce scope."""
        current_rows = [list(r) for r in rows]
        current_used = set(used_diffs)
        current_max = max(max(r) for r in current_rows)
        
        best_rows = [list(r) for r in current_rows]
        best_max = current_max
        
        T = 5.0
        T_min = 0.1
        cooling = 0.999
        
        def get_scope(rs):
            return max(max(r) for r in rs)
        
        while time.time() < time_limit:
            if T < T_min:
                T = 2.0
                cooling = 0.9995
            
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            p_idx = random.randint(1, len(row) - 1)
            
            old_val = row[p_idx]
            
            # Remove old_val
            diffs_to_remove = set()
            for a in row:
                if a != old_val:
                    diffs_to_remove.add(abs(a - old_val))
            
            current_used -= diffs_to_remove
            row.pop(p_idx)
            
            # Try to move to a new position
            # Search range: slightly wider than current max to allow moves, but prefer smaller
            max_search = min(best_max + 10, 250)
            
            valid_move = False
            new_val = None
            new_r_idx = r_idx
            
            if random.random() < 0.9:
                # Move within same row
                for _ in range(20): # Try a few random candidates
                    cand = random.randint(1, max_search)
                    if cand in row:
                        continue
                    
                    valid = True
                    new_diffs = set()
                    for a in row:
                        d = abs(cand - a)
                        if d in current_used:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Check if it maintains sorted order (we will re-sort later, but let's just insert)
                        # Actually, we just need to check conflicts.
                        pos = 0
                        while pos < len(row) and row[pos] < cand:
                            pos += 1
                        row.insert(pos, cand)
                        current_used |= new_diffs
                        new_val = cand
                        valid_move = True
                        break
            else:
                # Move to another row
                new_r_idx = random.randint(0, n_rows - 1)
                while new_r_idx == r_idx:
                    new_r_idx = random.randint(0, n_rows - 1)
                
                target_row = current_rows[new_r_idx]
                cand = old_val
                if cand in target_row:
                    cand = random.randint(1, max_search)
                    if cand in target_row:
                        # Revert
                        pos = 0
                        while pos < len(row) and row[pos] < old_val:
                            pos += 1
                        row.insert(pos, old_val)
                        current_used |= diffs_to_remove
                        continue
                
                valid = True
                new_diffs = set()
                for a in target_row:
                    d = abs(cand - a)
                    if d in current_used:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    pos = 0
                    while pos < len(target_row) and target_row[pos] < cand:
                        pos += 1
                    target_row.insert(pos, cand)
                    current_used |= new_diffs
                    new_val = cand
                    valid_move = True
            
            if not valid_move:
                # Revert
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                current_used |= diffs_to_remove
                continue
            
            new_max = get_scope(current_rows)
            delta = new_max - current_max
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in current_rows]
            else:
                # Revert
                if new_r_idx != r_idx:
                    target_row = current_rows[new_r_idx]
                    idx_new = target_row.index(new_val)
                    target_row.pop(idx_new)
                    for a in target_row:
                        d = abs(a - new_val)
                        current_used.discard(d)
                else:
                    idx_new = row.index(new_val)
                    row.pop(idx_new)
                    for a in row:
                        d = abs(a - new_val)
                        current_used.discard(d)
                
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                current_used |= diffs_to_remove
            
            T *= cooling
        
        return best_rows, best_max

    # --- Main Logic ---

    # 1. Greedy Baseline
    rows = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    # 2. Backtracking Search for Lower Scope
    # Try to find a solution with scope < best_max
    # We will try decreasing target scopes
    rng = random.Random(42)
    
    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1
    
    # Start slightly below greedy scope
    current_target = best_max - 1
    min_target = max(lower_bound, best_max - 20) # Don't go too low initially
    
    while current_target >= min_target and time.time() < deadline - 20:
        # Try a few times with different random seeds
        found = False
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            
            sol = backtracking_search(n_rows, k, current_target, rng, time.time() + 10)
            if sol:
                # Validate
                if is_valid(sol):
                    sol_max = max(max(r) for r in sol)
                    if sol_max < best_max:
                        best_max = sol_max
                        best_rows = [r[:] for r in sol]
                        found = True
                        break
        
        if found:
            # If we found a better solution, try to go even lower
            current_target = best_max - 1
            if current_target < min_target:
                break
        else:
            # If not found, lower the target
            current_target -= 1
            if current_target < min_target:
                break

    # 3. Local Improvement on the Best Found Solution
    used_diffs = get_used_diffs(best_rows)
    
    # Compaction
    compacted_rows, compacted_used = local_compact(best_rows, used_diffs, max_iterations=1000, time_limit=deadline - 10)
    compacted_max = max(max(r) for r in compacted_rows)
    
    if compacted_max < best_max:
        best_max = compacted_max
        best_rows = compacted_rows
        used_diffs = compacted_used

    # SA Polish
    if time.time() < deadline - 5:
        polished_rows, polished_max = sa_polish(best_rows, used_diffs, deadline - 2)
        if polished_max < best_max:
            best_max = polished_max
            best_rows = polished_rows

    # Final Validation
    if not is_valid(best_rows):
        # Fallback to greedy if something went wrong
        best_rows = greedy_construct(n_rows, k)

    return best_rows
