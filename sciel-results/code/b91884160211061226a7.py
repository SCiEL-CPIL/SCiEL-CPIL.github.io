import time
import random
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
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
    
    def greedy_construct(seed_val, max_scope=None):
        """Construct rows using greedy with a compact strategy."""
        if max_scope is None:
            max_scope = lb + 100
        random.seed(seed_val)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            
            while len(row) < k + 1:
                last = row[-1]
                found = False
                window = 20 + len(row) * 8
                search_end = min(last + window, max_scope)
                
                candidates = list(range(last + 1, search_end + 1))
                random.shuffle(candidates)
                
                for x in candidates:
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
                
                if not found:
                    # Expand search
                    for x in range(last + 1, max_scope + 1):
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
                    if not found:
                        return None, float('inf')
            
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope
    
    # 1. Get initial solution
    best_rows = None
    best_scope = float('inf')
    
    for s in range(50):
        if time.time() > deadline - 60:
            break
        rows, scope = greedy_construct(s)
        if rows and scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    if best_rows is None:
        # Fallback: Simple greedy with large search space
        used = set()
        best_rows = []
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
            best_rows.append(row)
        best_scope = max(max(r) for r in best_rows)
        
        # Ensure fallback is valid (it should be)
        if not is_valid(best_rows):
             pass

    # 2. Simulated Annealing
    current_rows = [list(r) for r in best_rows]
    
    # Build diff_counts
    diff_counts = {}
    current_conflict_count = 0
    
    def add_diff(d):
        nonlocal current_conflict_count
        if d in diff_counts:
            if diff_counts[d] == 1:
                current_conflict_count += 1
            diff_counts[d] += 1
        else:
            diff_counts[d] = 1
            
    def remove_diff(d):
        nonlocal current_conflict_count
        if d in diff_counts:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
            elif diff_counts[d] == 1:
                current_conflict_count -= 1

    for r in current_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                add_diff(d)
                
    current_scope = max(max(r) for r in current_rows)
    current_cost = current_scope + 1000 * current_conflict_count
    best_cost = current_cost
    best_rows_sa = [list(r) for r in current_rows]
    best_scope_sa = current_scope
    
    T = 15.0
    T_min = 0.1
    cooling = 0.9995
    
    steps = 0
    while time.time() < deadline - 10:
        if T < T_min:
            T = 5.0  # Reheat
            cooling = 0.9997
            if steps > 150000:
                break
        
        # Pick a random element to modify
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        p_idx = random.randint(1, len(row) - 1)  # Don't move 0
        
        old_val = row[p_idx]
        
        # Remove old_val's contributions
        diffs_to_remove = []
        for i in range(len(row)):
            if i != p_idx:
                diffs_to_remove.append(abs(row[i] - old_val))
        
        for d in diffs_to_remove:
            remove_diff(d)
        
        row.pop(p_idx)
        
        # Try to find a new valid value
        new_val = None
        new_r_idx = r_idx
        
        if random.random() < 0.7:
            # Value change in same row
            max_search = min(best_scope_sa + 30, 250)
            found = False
            for _ in range(8):
                cand = random.randint(1, max_search)
                if cand in row:
                    continue
                
                valid = True
                new_diffs = []
                for a in row:
                    d = abs(cand - a)
                    if d in diff_counts:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    pos = 0
                    while pos < len(row) and row[pos] < cand:
                        pos += 1
                    row.insert(pos, cand)
                    for d in new_diffs:
                        add_diff(d)
                    new_val = cand
                    found = True
                    break
            
            if not found:
                # Revert
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue
        else:
            # Move to another row
            new_r_idx = random.randint(0, n_rows - 1)
            while new_r_idx == r_idx:
                new_r_idx = random.randint(0, n_rows - 1)
            
            cand = old_val
            target_row = current_rows[new_r_idx]
            
            if cand in target_row:
                cand = random.randint(1, best_scope_sa + 20)
                if cand in target_row:
                    # Revert
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    for d in diffs_to_remove:
                        add_diff(d)
                    continue
            
            valid = True
            new_diffs = []
            for a in target_row:
                d = abs(cand - a)
                if d in diff_counts:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                pos = 0
                while pos < len(target_row) and target_row[pos] < cand:
                    pos += 1
                target_row.insert(pos, cand)
                for d in new_diffs:
                    add_diff(d)
                new_val = cand
            else:
                # Revert
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue
        
        # Calculate new cost
        new_scope = max(max(r) for r in current_rows)
        new_cost = new_scope + 1000 * current_conflict_count
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_cost = new_cost
            current_scope = new_scope
            
            # Only update best if the state is valid
            if is_valid(current_rows):
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope_sa = current_scope
                    best_rows_sa = [list(r) for r in current_rows]
        else:
            # Revert
            if new_r_idx != r_idx:
                target_row = current_rows[new_r_idx]
                idx_new = target_row.index(new_val)
                target_row.pop(idx_new)
                for a in target_row:
                    remove_diff(abs(a - new_val))
            else:
                row = current_rows[r_idx]
                idx_new = row.index(new_val)
                row.pop(idx_new)
                for a in row:
                    remove_diff(abs(a - new_val))
            
            # Re-insert old_val
            row = current_rows[r_idx]
            pos = 0
            while pos < len(row) and row[pos] < old_val:
                pos += 1
            row.insert(pos, old_val)
            for d in diffs_to_remove:
                add_diff(d)
        
        T *= cooling
        steps += 1
    
    # Final compaction
    final_rows = [list(r) for r in best_rows_sa]
    used_set = get_used_diffs(final_rows)
    final_scope = max(max(r) for r in final_rows)
    
    # Try to reduce scope
    for _ in range(300):
        if time.time() > deadline - 3:
            break
        row_idx = random.randint(0, n_rows - 1)
        row = final_rows[row_idx]
        if len(row) < 2:
            continue
        last = row[-1]
        if last <= 1:
            continue
        
        removed = row.pop()
        for a in row:
            used_set.discard(removed - a)
        
        found = False
        for x in range(last - 1, 0, -1):
            if x in row:
                continue
            valid = True
            for a in row:
                d = abs(x - a)
                if d == 0 or d in used_set:
                    valid = False
                    break
            if valid:
                pos = 0
                while pos < len(row) and row[pos] < x:
                    pos += 1
                row.insert(pos, x)
                for a in row:
                    if a != x:
                        used_set.add(abs(x - a))
                found = True
                break
        
        if not found:
            row.append(removed)
            row.sort()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_set.add(row[j] - row[i])
    
    final_scope = max(max(r) for r in final_rows)
    
    # Validate final result
    if is_valid(final_rows) and final_scope < best_scope_sa:
        best_rows_sa = final_rows
        best_scope_sa = final_scope
        
    # Final safety check: if best_rows_sa is invalid, use best_rows (greedy)
    if not is_valid(best_rows_sa):
        if is_valid(best_rows):
            best_rows_sa = best_rows
        else:
            # Emergency fallback: generate a valid set, even if large
            used = set()
            best_rows_sa = []
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
                best_rows_sa.append(row)

    return best_rows_sa
