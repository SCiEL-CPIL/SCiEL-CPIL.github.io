import time
import random
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin before 150s kill
    
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

    def compaction_pass(rows, used_set, max_iters=500):
        """Aggressively reduce scope by trying to move each element to a smaller value."""
        final_scope = max(max(r) for r in rows)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False
            for row_idx in range(n_rows):
                row = rows[row_idx]
                for p_idx in range(len(row) - 1, 0, -1):  # from right to left
                    old_val = row[p_idx]
                    if old_val <= 1:
                        continue
                    # Try to replace old_val with a smaller value
                    for x in range(old_val - 1, 0, -1):
                        if x in row:
                            continue
                        valid = True
                        for i in range(len(row)):
                            if i == p_idx:
                                continue
                            d = abs(row[i] - x)
                            if d == 0 or d in used_set:
                                valid = False
                                break
                        if valid:
                            # Remove old diffs
                            for i in range(len(row)):
                                if i != p_idx:
                                    used_set.discard(abs(row[i] - old_val))
                            # Add new diffs
                            for i in range(len(row)):
                                if i != p_idx:
                                    used_set.add(abs(row[i] - x))
                            row[p_idx] = x
                            row.sort()
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return rows
    
    # 1. Get initial solution
    best_rows = None
    best_scope = float('inf')
    
    # Quick deterministic baseline first
    used_base = set()
    base_rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                x += 1
                valid = True
                for a in row:
                    if (x - a) in used_base:
                        valid = False
                        break
                if valid:
                    break
            row.append(x)
            used_base.update({x - a for a in row if a != x})
        base_rows.append(row)
    base_scope = max(max(r) for r in base_rows)
    best_rows = [list(r) for r in base_rows]
    best_scope = base_scope
    
    # Then try randomized greedy for potentially better starts
    for s in range(30):
        if time.time() > deadline - 80:
            break
        rows, scope = greedy_construct(s)
        if rows and scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]

    # 2. Simulated Annealing with bitset for fast diff checking
    current_rows = [list(r) for r in best_rows]
    
    # Build diff bitset for fast membership
    max_possible = best_scope + 50
    diff_bitset = 0  # Use a Python int as a bitset
    
    def diff_in_bitset(d):
        return (diff_bitset >> d) & 1
    
    def add_to_bitset(d):
        nonlocal diff_bitset
        diff_bitset |= (1 << d)
    
    def remove_from_bitset(d):
        nonlocal diff_bitset
        diff_bitset &= ~(1 << d)
    
    # Initialize bitset
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                add_to_bitset(r[j] - r[i])
    
    current_scope = max(max(r) for r in current_rows)
    best_scope_sa = current_scope
    best_rows_sa = [list(r) for r in current_rows]
    
    T = 10.0
    T_min = 0.1
    cooling = 0.9998
    
    steps = 0
    while time.time() < deadline - 15:
        if T < T_min:
            T = 5.0  # Reheat
            cooling = 0.99985
            if steps > 200000:
                break
        
        # Pick a random element to modify
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        p_idx = random.randint(1, len(row) - 1)  # Don't move 0
        
        old_val = row[p_idx]
        
        # Remove old_val's contributions from bitset
        for i in range(len(row)):
            if i != p_idx:
                remove_from_bitset(abs(row[i] - old_val))
        
        row.pop(p_idx)
        
        # Try to find a new valid value
        new_val = None
        new_r_idx = r_idx
        success = False
        
        if random.random() < 0.7:
            # Value change in same row
            max_search = min(best_scope_sa + 20, 250)
            for _ in range(5):
                cand = random.randint(1, max_search)
                if cand in row:
                    continue
                
                valid = True
                new_diffs = []
                for a in row:
                    d = abs(cand - a)
                    if d == 0 or diff_in_bitset(d):
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    pos = 0
                    while pos < len(row) and row[pos] < cand:
                        pos += 1
                    row.insert(pos, cand)
                    for d in new_diffs:
                        add_to_bitset(d)
                    new_val = cand
                    success = True
                    break
            
            if not success:
                # Revert
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for i in range(len(row)):
                    if i != p_idx:
                        add_to_bitset(abs(row[i] - old_val))
                continue
        else:
            # Move to another row
            new_r_idx = random.randint(0, n_rows - 1)
            while new_r_idx == r_idx:
                new_r_idx = random.randint(0, n_rows - 1)
            
            cand = old_val
            target_row = current_rows[new_r_idx]
            
            if cand in target_row:
                cand = random.randint(1, best_scope_sa + 15)
                if cand in target_row:
                    # Revert
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    for i in range(len(row)):
                        if i != p_idx:
                            add_to_bitset(abs(row[i] - old_val))
                    continue
            
            valid = True
            new_diffs = []
            for a in target_row:
                d = abs(cand - a)
                if d == 0 or diff_in_bitset(d):
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                pos = 0
                while pos < len(target_row) and target_row[pos] < cand:
                    pos += 1
                target_row.insert(pos, cand)
                for d in new_diffs:
                    add_to_bitset(d)
                new_val = cand
                success = True
            else:
                # Revert
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for i in range(len(row)):
                    if i != p_idx:
                        add_to_bitset(abs(row[i] - old_val))
                continue
        
        # Calculate new scope
        new_scope = max(max(r) for r in current_rows)
        
        # Accept if better scope, or with probability based on temperature
        if new_scope <= best_scope_sa:
            # Always accept if not worse
            if new_scope < best_scope_sa:
                best_scope_sa = new_scope
                best_rows_sa = [list(r) for r in current_rows]
        else:
            # Accept with probability exp(-(new_scope - best_scope_sa) / T)
            delta = new_scope - best_scope_sa
            if random.random() >= math.exp(-delta / T):
                # Revert
                if new_r_idx != r_idx:
                    target_row = current_rows[new_r_idx]
                    idx_new = target_row.index(new_val)
                    target_row.pop(idx_new)
                    for a in target_row:
                        remove_from_bitset(abs(a - new_val))
                else:
                    row = current_rows[r_idx]
                    idx_new = row.index(new_val)
                    row.pop(idx_new)
                    for a in row:
                        remove_from_bitset(abs(a - new_val))
                
                # Re-insert old_val
                row = current_rows[r_idx]
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for i in range(len(row)):
                    if i != p_idx:
                        add_to_bitset(abs(row[i] - old_val))
        
        T *= cooling
        steps += 1
    
    # Final compaction
    final_rows = [list(r) for r in best_rows_sa]
    used_set = get_used_diffs(final_rows)
    final_scope = max(max(r) for r in final_rows)
    
    # Aggressive compaction: try to move each element to a smaller value
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for row_idx in range(n_rows):
            row = final_rows[row_idx]
            for p_idx in range(len(row) - 1, 0, -1):  # from right to left
                old_val = row[p_idx]
                if old_val <= 1:
                    continue
                # Try to replace old_val with a smaller value
                for x in range(old_val - 1, 0, -1):
                    if x in row:
                        continue
                    valid = True
                    for i in range(len(row)):
                        if i == p_idx:
                            continue
                        d = abs(row[i] - x)
                        if d == 0 or d in used_set:
                            valid = False
                            break
                    if valid:
                        # Remove old diffs
                        for i in range(len(row)):
                            if i != p_idx:
                                used_set.discard(abs(row[i] - old_val))
                        # Add new diffs
                        for i in range(len(row)):
                            if i != p_idx:
                                used_set.add(abs(row[i] - x))
                        row[p_idx] = x
                        row.sort()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
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
