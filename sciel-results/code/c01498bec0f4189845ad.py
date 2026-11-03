import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    rows, M_init = greedy_construct(n_rows, k)
    best_rows = rows
    best_M = M_init
    
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    M_min = total_diffs
    
    for M_target in range(M_init - 1, M_min - 1, -1):
        if time.time() > deadline - 5.0:
            break
        
        found_rows = None
        max_restarts = 20
        
        for _ in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
            
            res = backtrack_search(n_rows, k, M_target, deadline)
            if res is not None:
                found_rows = res
                break
        
        if found_rows is not None:
            best_rows = found_rows
            best_M = M_target
            
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
                used.update(new_diffs)
        rows.append(row)
    max_val = max(max(r) for r in rows)
    return rows, max_val

def backtrack_search(n_rows, k, M, deadline):
    if time.time() > deadline:
        return None
        
    diffs_per_row = (k + 1) * k // 2
    rows = []
    
    def dfs(row_idx, current_row, used_mask):
        if time.time() > deadline:
            return None
            
        if row_idx == n_rows:
            return rows.copy()
        
        if len(current_row) == k + 1:
            rows.append(current_row)
            res = dfs(row_idx + 1, [0], used_mask)
            if res is not None:
                return res
            rows.pop()
            return None
        
        last = current_row[-1]
        candidates = []
        
        for x in range(last + 1, M + 1):
            mask = 0
            valid = True
            for a in current_row:
                d = x - a
                bit = 1 << d
                if used_mask & bit:
                    valid = False
                    break
                mask |= bit
            if valid:
                candidates.append((x, mask))
        
        random.shuffle(candidates)
        
        for x, mask in candidates:
            new_used_mask = used_mask | mask
            
            L = len(current_row)
            rem_elems = k - L
            if rem_elems > 0:
                current_row_remaining_diffs = (2 * L + rem_elems + 1) * rem_elems // 2
            else:
                current_row_remaining_diffs = 0
            
            future_rows_diffs = (n_rows - row_idx - 1) * diffs_per_row
            
            total_needed = new_used_mask.bit_count() + current_row_remaining_diffs + future_rows_diffs
            
            if total_needed > M:
                continue
            
            res = dfs(row_idx, current_row + [x], new_used_mask)
            if res is not None:
                return res
        
        return None

    return dfs(0, [0], 0)
