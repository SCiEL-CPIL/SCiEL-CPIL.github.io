import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_scope = float('inf')
    best_rows = None
    
    while time.time() < deadline:
        rows = randomized_greedy(n_rows, k)
        if rows is None:
            continue
            
        scope = max(max(row) for row in rows)
        rows, scope = local_shrink(rows, scope)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            
    if best_rows is None:
        best_rows = deterministic_greedy(n_rows, k)
        
    return best_rows

def deterministic_greedy(n_rows, k):
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

def randomized_greedy(n_rows, k):
    used = set()
    rows = []
    num_candidates = random.randint(3, 15)
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            valid_candidates = []
            start_x = row[-1]
            
            while len(valid_candidates) < num_candidates:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    valid_candidates.append(x)
                if x > 5000:
                    break
                    
            if not valid_candidates:
                return None
            
            chosen_x = random.choice(valid_candidates)
            row.append(chosen_x)
            used |= {chosen_x - a for a in row[:-1]}
            
        rows.append(row)
    return rows

def local_shrink(rows, scope):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used.add(row[j] - row[i])
                
    improved = True
    max_iter = 100
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for r_idx in range(len(rows)):
            row = rows[r_idx]
            if len(row) < 2:
                continue
            last_idx = len(row) - 1
            old_val = row[last_idx]
            min_val = row[last_idx - 1] + 1
            
            if min_val >= old_val:
                continue
                
            best_new_val = None
            # Try to find the smallest valid value
            for new_val in range(min_val, old_val):
                prev_elements = row[:last_idx]
                old_diffs = {old_val - a for a in prev_elements}
                new_diffs = {new_val - a for a in prev_elements}
                
                conflict = False
                for d in new_diffs:
                    if d in used and d not in old_diffs:
                        conflict = True
                        break
                        
                if not conflict:
                    best_new_val = new_val
                    break
            
            if best_new_val is not None:
                prev_elements = row[:last_idx]
                old_diffs = {old_val - a for a in prev_elements}
                new_diffs = {best_new_val - a for a in prev_elements}
                
                used -= old_diffs
                used |= new_diffs
                row[last_idx] = best_new_val
                improved = True
                
    return rows, max(max(r) for r in rows)
