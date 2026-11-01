"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_rows = None
    best_scope = float('inf')
    
    # 1. Deterministic Greedy Initialization
    rows = _greedy_construct(n_rows, k)
    if rows:
        best_rows = [r[:] for r in rows]
        best_scope = max(max(r) for r in rows)
        
    # 2. Scope-Driven Local Search with Restarts
    # Strategy: Focus on reducing the global maximum. If stuck, perturb other elements.
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    no_improve_count = 0
    max_no_improve = 20 # Restart after 20 failed passes
    
    while time.time() < deadline:
        improved, new_rows, new_scope = _scope_local_search(current_rows, k, deadline)
        
        if improved:
            current_rows = new_rows
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
            no_improve_count = 0
            if best_scope <= 105:
                break
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Randomized restart
                if time.time() > deadline - 5:
                    break
                random.seed(int(time.time() * 1000000) % (2**32))
                rand_rows = _randomized_greedy(n_rows, k, deadline)
                if rand_rows:
                    # Quick refine of random start
                    imp_r, ref_rows, ref_scope = _scope_local_search(rand_rows, k, deadline)
                    if ref_scope < best_scope:
                        best_scope = ref_scope
                        best_rows = [r[:] for r in ref_rows]
                        current_rows = ref_rows
                        current_scope = ref_scope
                        no_improve_count = 0
                    else:
                        # Keep current best, but maybe mix? 
                        # For now, just reset the counter and continue with best
                        current_rows = [r[:] for r in best_rows]
                        current_scope = best_scope
                        no_improve_count = 0
                else:
                    break
                    
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

def _randomized_greedy(n_rows, k, deadline):
    used, rows = set(), []
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        row, x = [0], 0
        while len(row) < k + 1:
            if time.time() > deadline:
                return None
            # Try candidates in a biased random order
            # Start from x+1, look up to x+100
            candidates = list(range(x + 1, x + 100))
            random.shuffle(candidates)
            # Bias towards smaller values
            candidates.sort(key=lambda c: c + random.random() * 20)
            
            found = False
            for cand in candidates:
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
                    used |= new
                    x = cand
                    found = True
                    break
            if not found:
                # If not found in range, try linear scan up to x+500
                for cand in range(x + 100, x + 500):
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
            if not found:
                return None
        rows.append(row)
    return rows

def _scope_local_search(rows, k, deadline):
    """
    Attempts to reduce the global scope (max entry) of the rows.
    Focuses moves on the current maximum element. If stuck, tries to move other elements.
    """
    curr_rows = [r[:] for r in rows]
    n_rows = len(curr_rows)
    k_plus_1 = k + 1
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    global_diffs = set()
    for row in curr_rows:
        global_diffs |= get_row_diffs(row)
        
    current_scope = max(max(r) for r in curr_rows)
    improved = False
    
    # Max iterations to prevent infinite loops in a single call
    max_iters = 500 
    iters = 0
    
    while time.time() < deadline and iters < max_iters:
        iters += 1
        
        # Find the position of the global maximum
        max_val = current_scope
        # Find all indices with this max value
        max_indices = []
        for i in range(n_rows):
            for j in range(1, k_plus_1):
                if curr_rows[i][j] == max_val:
                    max_indices.append((i, j))
        
        # Try to move one of the max elements down
        move_found = False
        if max_indices:
            # Randomly pick one of the max elements to try first
            random.shuffle(max_indices)
            for i, j in max_indices:
                row = curr_rows[i]
                current_val = row[j]
                min_val = row[j-1] + 1
                max_val_constraint = row[j+1] - 1 if j + 1 < k_plus_1 else current_val - 1
                
                old_diffs = get_row_diffs(row)
                global_diffs -= old_diffs
                
                # Scan down from current_val - 1
                best_candidate = None
                for cand in range(current_val - 1, min_val - 1, -1):
                    if cand >= max_val_constraint:
                        continue
                    
                    new_row = row[:]
                    new_row[j] = cand
                    new_diffs = get_row_diffs(new_row)
                    
                    if len(new_diffs) != k * (k + 1) // 2:
                        continue
                    if new_diffs & global_diffs:
                        continue
                    
                    best_candidate = cand
                    break # Found a valid smaller value for the max
                
                if best_candidate is not None:
                    row[j] = best_candidate
                    global_diffs |= get_row_diffs(row)
                    current_scope = max(max(r) for r in curr_rows)
                    improved = True
                    move_found = True
                    break
                else:
                    global_diffs |= old_diffs
        
        if move_found:
            continue
        
        # If we couldn't move the max, try to move a random other element to "unblock"
        # This is a heuristic to escape local minima where the max is trapped.
        # Pick a random element (not max) and try to reduce it.
        all_elements = [(i, j) for i in range(n_rows) for j in range(1, k_plus_1)]
        random.shuffle(all_elements)
        
        for i, j in all_elements:
            row = curr_rows[i]
            current_val = row[j]
            
            # Only try to reduce
            if current_val <= 1:
                continue
                
            min_val = row[j-1] + 1
            max_val_constraint = row[j+1] - 1 if j + 1 < k_plus_1 else current_val - 1
            
            old_diffs = get_row_diffs(row)
            global_diffs -= old_diffs
            
            best_candidate = None
            # Try to find a smaller value
            # To save time, maybe just try a few random candidates or scan down a bit?
            # Scanning down is safer for "tightening".
            for cand in range(current_val - 1, min_val - 1, -1):
                if cand >= max_val_constraint:
                    continue
                
                new_row = row[:]
                new_row[j] = cand
                new_diffs = get_row_diffs(new_row)
                
                if len(new_diffs) != k * (k + 1) // 2:
                    continue
                if new_diffs & global_diffs:
                    continue
                
                best_candidate = cand
                break
            
            if best_candidate is not None:
                row[j] = best_candidate
                global_diffs |= get_row_diffs(row)
                # Scope might not change, but structure did
                improved = True # We made a move, so it's "improved" in the sense of progress
                break
            else:
                global_diffs |= old_diffs
        
        if not move_found and not improved:
            # No moves possible
            break
        elif not move_found:
            # We moved a non-max element. Check if scope changed?
            # It shouldn't have increased. It might have decreased if that element was max? 
            # No, we excluded max indices in the first block. 
            # So scope is same. But we need to reset 'improved' logic?
            # Actually, if we moved a non-max element, we should continue to see if max can now move.
            # But 'improved' flag here indicates "did we make a move in this iteration".
            # We return improved=True if *any* move was made.
            # The caller checks if scope decreased to update best.
            pass

    return improved, curr_rows, current_scope

# EVOLVE-BLOCK-END
