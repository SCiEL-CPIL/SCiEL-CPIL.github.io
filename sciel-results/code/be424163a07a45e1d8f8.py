import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer
    
    best_set = None
    best_max = float('inf')
    
    # 1. Generate initial candidates using randomized greedy
    # We try a few variations to escape the deterministic greedy trap.
    num_candidates = 20
    
    for i in range(num_candidates):
        if time.time() > deadline - 5.0: # Leave 5s for final polish/return
            break
            
        # Randomized greedy
        rows = _greedy_build(n_rows, k, randomize=(i > 0))
        
        current_max = _get_max(rows)
        if current_max < best_max:
            best_max = current_max
            best_set = [row[:] for row in rows]
            
        # Local search to improve this candidate
        improved_rows, improved_max = _local_search(rows, current_max, deadline)
        
        if improved_max < best_max:
            best_max = improved_max
            best_set = [row[:] for row in improved_rows]

    # If we still have time, do a final polish on the best set
    if best_set is not None and time.time() < deadline - 2.0:
        final_rows, final_max = _local_search(best_set, best_max, deadline, polish=True)
        if final_max < best_max:
            best_max = final_max
            best_set = final_rows

    if best_set is None:
        # Fallback: pure deterministic greedy
        best_set = _greedy_build(n_rows, k, randomize=False)
        
    return best_set

def _greedy_build(n_rows, k, randomize=False):
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                if randomize:
                    # With small probability, skip this valid x and try a larger one
                    # to create structural diversity.
                    if random.random() < 0.15:
                        # Try x+1, x+2...
                        # Just try one step up
                        x_next = x + 1
                        new_diffs_next = {x_next - a for a in row}
                        if not new_diffs_next & used:
                            x = x_next
                            new_diffs = new_diffs_next
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _get_max(rows):
    return max(max(row) for row in rows)

def _local_search(rows, current_max, deadline, polish=False):
    rows = [row[:] for row in rows]
    
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used.add(row[j] - row[i])
                
    n_rows = len(rows)
    
    iterations = 0
    max_iterations = 100000 if not polish else 500000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Find the global maximum
        global_max = _get_max(rows)
        
        # Find an element equal to global_max
        target_row_idx = -1
        target_elem_idx = -1
        
        for i, row in enumerate(rows):
            if global_max in row:
                target_row_idx = i
                target_elem_idx = row.index(global_max)
                break
        
        if target_row_idx == -1:
            break
            
        row = rows[target_row_idx]
        max_val = row[target_elem_idx]
        
        reduced = False
        diffs_to_remove = set()
        for a in row:
            if a != max_val:
                diffs_to_remove.add(abs(max_val - a))
        
        used_without_max = used - diffs_to_remove
        row_set = set(row)
        
        found_v = None
        for v in range(1, max_val):
            if v in row_set:
                continue
                
            new_diffs = set()
            valid = True
            for a in row:
                if a != max_val:
                    d = abs(v - a)
                    if d in used_without_max:
                        valid = False
                        break
                    new_diffs.add(d)
            
            if valid:
                found_v = v
                break
                
        if found_v is not None:
            used -= diffs_to_remove
            used |= new_diffs
            row[target_elem_idx] = found_v
            row.sort()
            reduced = True
        else:
            # Try a swap
            if not polish or random.random() < 0.5:
                if _try_swap(rows, used, global_max, deadline):
                    pass
                else:
                    # If stuck, we might want to break out of this local search
                    # to let the outer loop try another candidate.
                    # But we are in a while loop.
                    # Let's just continue.
                    pass

    return rows, _get_max(rows)

def _try_swap(rows, used, current_global_max, deadline):
    n_rows = len(rows)
    if n_rows < 2:
        return False
        
    r1, r2 = random.sample(range(n_rows), 2)
    
    row1 = rows[r1]
    row2 = rows[r2]
    
    if len(row1) < 2 or len(row2) < 2:
        return False
        
    idx1 = random.randint(1, len(row1)-1)
    idx2 = random.randint(1, len(row2)-1)
    
    v1 = row1[idx1]
    v2 = row2[idx2]
    
    if v1 == v2:
        return False
        
    d1_remove = set()
    for a in row1:
        if a != v1:
            d1_remove.add(abs(v1 - a))
            
    d2_remove = set()
    for a in row2:
        if a != v2:
            d2_remove.add(abs(v2 - a))
            
    used_temp = used - d1_remove - d2_remove
    
    d1_add = set()
    for a in row1:
        if a != v1:
            d = abs(v2 - a)
            if d in used_temp:
                return False
            if d in d1_add:
                return False
            d1_add.add(d)
            
    d2_add = set()
    for a in row2:
        if a != v2:
            d = abs(v1 - a)
            if d in used_temp:
                return False
            if d in d1_add:
                return False
            d2_add.add(d)
            
    # Perform swap on used
    used -= d1_remove
    used -= d2_remove
    used |= d1_add
    used |= d2_add
    
    # Perform swap on rows
    row1[idx1] = v2
    row2[idx2] = v1
    
    row1.sort()
    row2.sort()
    
    # Check if global max increased
    new_global_max = _get_max(rows)
    if new_global_max > current_global_max:
        # Revert
        row1[idx1] = v1
        row2[idx2] = v2
        row1.sort()
        row2.sort()
        used -= d1_add
        used -= d2_add
        used |= d1_remove
        used |= d2_remove
        return False
        
    return True

# EVOLVE-BLOCK-END
