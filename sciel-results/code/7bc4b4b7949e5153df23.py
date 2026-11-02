import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def greedy_construct(seed_val):
        """Construct rows using greedy with a compact strategy."""
        random.seed(seed_val)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                
                # Try to find the smallest x > last such that all (x - a) for a in row are not in used
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    x += 1
                
                if not found:
                    return None, float('inf')
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope

    # 1. Get a valid starting point using a few randomized greedy restarts
    best_initial_rows = None
    best_initial_scope = float('inf')
    
    # Do a few quick restarts to find a good starting point
    for seed in range(5):
        if time.time() > deadline - 50: # Leave time for local search
            break
        rows, scope = greedy_construct(seed)
        if rows is not None and scope < best_initial_scope:
            best_initial_scope = scope
            best_initial_rows = [list(r) for r in rows]
            
    if best_initial_rows is None:
        # Fallback: deterministic greedy
        rows, scope = greedy_construct(0)
        if rows is None:
            # Extremely sparse fallback
            used = set()
            best_initial_rows = []
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used:
                                valid = False
                                break
                        if valid:
                            break
                    row.append(x)
                    used.update({x - a for a in row if a != x})
                best_initial_rows.append(row)
            best_initial_scope = max(max(r) for r in best_initial_rows)
        else:
            best_initial_rows = [list(r) for r in rows]
            best_initial_scope = scope

    current_rows = [list(r) for r in best_initial_rows]
    current_used = get_used_diffs(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = best_initial_scope

    # 2. Local Search: Iterative Compaction
    # Strategy: Pick a random row, pick a random element (not 0), try to move it to a smaller value.
    # If valid, accept. This reduces the scope if the moved element was the max, or just improves density.
    
    def try_move_element(rows, used_set, r_idx, p_idx, new_val):
        """
        Attempt to move element at rows[r_idx][p_idx] to new_val.
        Returns True if successful and updates rows/used_set, False otherwise.
        """
        old_val = rows[r_idx][p_idx]
        if old_val == new_val:
            return True
            
        # Check if new_val is already in the row
        if new_val in rows[r_idx]:
            return False
        
        row = rows[r_idx]
        
        # 1. Remove old_val from row and used_set
        # Remove diffs involving old_val
        for a in row:
            if a != old_val:
                d = abs(a - old_val)
                used_set.discard(d)
        
        # Temporarily remove old_val from row
        rows[r_idx].pop(p_idx)
        
        # 2. Check if new_val is valid
        valid = True
        new_diffs = []
        for a in rows[r_idx]:
            d = abs(a - new_val)
            if d in used_set:
                valid = False
                break
            new_diffs.append(d)
            
        if valid:
            # 3. Insert new_val into correct position to keep sorted
            pos = 0
            while pos < len(rows[r_idx]) and rows[r_idx][pos] < new_val:
                pos += 1
            rows[r_idx].insert(pos, new_val)
            
            # 4. Add new diffs
            for d in new_diffs:
                used_set.add(d)
                
            return True
        else:
            # Revert: put old_val back
            pos = 0
            while pos < len(rows[r_idx]) and rows[r_idx][pos] < old_val:
                pos += 1
            rows[r_idx].insert(pos, old_val)
            
            # Re-add old diffs
            for a in rows[r_idx]:
                if a != old_val:
                    d = abs(a - old_val)
                    used_set.add(d)
                    
            return False

    # Main Local Search Loop
    while time.time() < deadline - 2:
        # Prioritize moving the global maximum element
        max_val = -1
        max_r_idx = -1
        max_p_idx = -1
        
        for r_idx in range(n_rows):
            for p_idx in range(len(current_rows[r_idx])):
                if current_rows[r_idx][p_idx] > max_val:
                    max_val = current_rows[r_idx][p_idx]
                    max_r_idx = r_idx
                    max_p_idx = p_idx
        
        # Try to move the max element
        if max_p_idx > 0: # Not 0
            old_val = current_rows[max_r_idx][max_p_idx]
            if old_val > 1:
                # Try to find a smaller valid value
                found = False
                # Try random candidates first for speed
                if old_val > 5:
                    candidates = random.sample(range(1, old_val), min(5, old_val - 1))
                else:
                    candidates = list(range(1, old_val))
                for cand in candidates:
                    if try_move_element(current_rows, current_used, max_r_idx, max_p_idx, cand):
                        found = True
                        break
                
                # If random failed, try linear scan downwards
                if not found:
                    for cand in range(old_val - 1, max(old_val - 10, 0), -1):
                        if try_move_element(current_rows, current_used, max_r_idx, max_p_idx, cand):
                            found = True
                            break

        # Also try random moves to escape local minima
        r_idx = random.randint(0, n_rows - 1)
        p_idx = random.randint(1, len(current_rows[r_idx]) - 1)
        old_val = current_rows[r_idx][p_idx]
        
        if old_val > 1:
            if old_val > 5:
                candidates = random.sample(range(1, old_val), min(5, old_val - 1))
            else:
                candidates = list(range(1, old_val))
            for cand in candidates:
                if try_move_element(current_rows, current_used, r_idx, p_idx, cand):
                    break

        # Update best
        new_scope = max(max(r) for r in current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [list(r) for r in current_rows]

    if best_rows is None:
        best_rows = initial_rows
        
    return best_rows
