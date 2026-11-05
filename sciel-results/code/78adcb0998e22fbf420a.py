import time
import random
import math
import sys

def _row_diffs_mask(row):
    """Return bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                if d in seen:
                    return False
                seen.add(d)
    return True

def _greedy_rebuild(used_mask, num_elements, rng, max_val_hint=200):
    """
    Rebuild a single row given a used_mask of differences from OTHER rows.
    Returns a row that minimizes the last element (greedy smallest valid).
    """
    row = [0]
    for _ in range(num_elements - 1):
        found = False
        start_val = row[-1] + 1
        
        # We search up to max_val_hint. 
        # If not found, we return None.
        search_limit = max_val_hint
        
        for val in range(start_val, search_limit + 1):
            valid = True
            local_diffs = []
            for prev in row:
                d = val - prev
                # d is always positive since val > row[-1] >= prev
                if (used_mask >> d) & 1:
                    valid = False
                    break
                if d in local_diffs:
                    valid = False
                    break
                local_diffs.append(d)
            
            if valid:
                row.append(val)
                found = True
                break
        
        if not found:
            return None
            
    return row

def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 145.0
    
    num_elements = k + 1
    rng = random.Random(42)
    
    # Phase 1: Initial Construction
    rows = []
    global_used_mask = 0
    
    for i in range(n_rows):
        row = _greedy_rebuild(global_used_mask, num_elements, rng, max_val_hint=500)
        if row is None:
            # Fallback: brute force
            row = [0]
            while len(row) < num_elements:
                found = False
                for v in range(row[-1]+1, 1000):
                    ok = True
                    local_diffs = set()
                    for prev in row:
                        d = v - prev
                        if (global_used_mask >> d) & 1:
                            ok = False
                            break
                        if d in local_diffs:
                            ok = False
                            break
                        local_diffs.add(d)
                    if ok:
                        row.append(v)
                        found = True
                        break
                if not found:
                    row.append(row[-1] + 100)
                    break
        
        rows.append(row)
        m = _row_diffs_mask(row)
        global_used_mask |= m
        
    current_rows = rows
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # Phase 2: Iterated Local Search with Row Rebuilds
    temp = 15.0
    cooling = 0.9995
    min_temp = 0.5
    stagnation = 0
    max_stagnation = 5000
    
    while time.time() < DEADLINE and temp > min_temp:
        if stagnation > max_stagnation:
            # Perturbation: randomly rebuild 2-3 rows with high hint
            for _ in range(3):
                idx = rng.randint(0, n_rows - 1)
                old_m = _row_diffs_mask(current_rows[idx])
                others = global_used_mask & ~old_m
                new_r = _greedy_rebuild(others, num_elements, rng, max_val_hint=current_scope + 100)
                if new_r:
                    current_rows[idx] = new_r
                    global_used_mask = (global_used_mask & ~old_m) | _row_diffs_mask(new_r)
            current_scope = _scope(current_rows)
            stagnation = 0
            temp = max(temp, 5.0)
            
        row_idx = rng.randint(0, n_rows - 1)
        old_row = current_rows[row_idx]
        old_row_mask = _row_diffs_mask(old_row)
        others_mask = global_used_mask & ~old_row_mask
        
        # Rebuild
        # Use a hint slightly above current scope to allow some flexibility
        new_row = _greedy_rebuild(others_mask, num_elements, rng, max_val_hint=current_scope + 20)
        
        if new_row is None:
            continue
            
        new_row_mask = _row_diffs_mask(new_row)
        
        if (new_row_mask & others_mask) != 0:
            continue
            
        temp_rows = current_rows[:]
        temp_rows[row_idx] = new_row
        new_scope = _scope(temp_rows)
        
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current_rows[row_idx] = new_row
            global_used_mask = (global_used_mask & ~old_row_mask) | new_row_mask
            current_scope = new_scope
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
            
        temp *= cooling
        
    if _is_valid(best_rows):
        return best_rows
    else:
        return current_rows

