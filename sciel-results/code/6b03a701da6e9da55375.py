import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_rows = greedy_construct(n_rows, k)
    if not best_rows:
        return []
    
    best_scope = max(max(row) for row in best_rows)
    
    total_diffs = n_rows * (k * (k + 1) // 2)
    lower_bound = total_diffs
    
    if best_scope == lower_bound:
        return best_rows

    while time.time() < deadline:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
            
        slice_deadline = time.time() + 2.0
        found_rows = backtrack_search(n_rows, k, target_scope, slice_deadline)
        
        if found_rows:
            scope = max(max(row) for row in found_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = found_rows
        else:
            # Optional: if we fail many times, we might be at the optimum.
            # But we don't know. Just keep trying.
            pass

    return best_rows

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def backtrack_search(n_rows, k, max_scope, deadline):
    used_diffs = 0
    rows = []
    
    def solve_idx(row_idx):
        if row_idx == n_rows:
            return True
        if time.time() > deadline:
            return False
        prev_row = rows[row_idx - 1] if row_idx > 0 else None
        return build_row_idx(0, [0], 0, used_diffs, 0, row_idx, prev_row)

    def build_row_idx(pos, current_row, last_val, used_diffs, local_mask, row_idx, prev_row):
        if pos == k:
            # Symmetry breaking
            if prev_row is not None:
                if current_row <= prev_row:
                    return False
            
            if (local_mask & used_diffs) != 0:
                return False
            
            rows.append(current_row)
            if solve_idx(row_idx + 1):
                return True
            rows.pop()
            return False
        
        if max_scope - last_val < k - pos:
            return False
            
        start = last_val + 1
        end = max_scope
        candidates = list(range(start, end + 1))
        random.shuffle(candidates)
        
        for v in candidates:
            new_local_mask = local_mask
            valid = True
            for u in current_row:
                d = v - u
                bit = 1 << d
                if (new_local_mask & bit) != 0:
                    valid = False
                    break
                new_local_mask |= bit
            
            if not valid:
                continue
                
            if (new_local_mask & used_diffs) != 0:
                continue
                
            current_row.append(v)
            if build_row_idx(pos + 1, current_row, v, used_diffs, new_local_mask, row_idx, prev_row):
                return True
            current_row.pop()
            
        return False

    if solve_idx(0):
        return rows
    return None
