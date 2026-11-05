import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin from 150s limit
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True
    
    # Greedy initial
    def deterministic_greedy():
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
    
    best_rows = deterministic_greedy()
    best_scope = get_scope(best_rows)
    
    # Phase 1: Backtracking search to find a good initial solution
    total_diffs = n_rows * k * (k+1) // 2
    node_count = [0]
    
    def backtrack(row_idx, used_diffs, current_rows, current_max):
        nonlocal best_rows, best_scope
        if time.time() > deadline:
            return
        node_count[0] += 1
        if node_count[0] > 500000:
            return
        if row_idx == n_rows:
            if len(used_diffs) == total_diffs:
                if current_max < best_scope:
                    best_scope = current_max
                    best_rows = [r[:] for r in current_rows]
            return
        
        row = [0]
        def build(pos, row, row_diffs, row_max):
            nonlocal best_scope
            if time.time() > deadline:
                return
            node_count[0] += 1
            if node_count[0] > 500000:
                return
            if pos == k + 1:
                if row_diffs & used_diffs:
                    return
                new_used = used_diffs | row_diffs
                current_rows.append(row[:])
                backtrack(row_idx + 1, new_used, current_rows, max(current_max, row_max))
                current_rows.pop()
                return
            
            start_x = row[-1] + 1
            for x in range(start_x, best_scope + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & row_diffs or new_diffs & used_diffs:
                    continue
                row.append(x)
                build(pos + 1, row, row_diffs | new_diffs, max(row_max, x))
                row.pop()
        
        build(1, row, set(), 0)
    
    # Run backtracking with time limit
    bt_deadline = start_time + 30.0  # Give 30s to backtracking
    old_deadline = deadline
    deadline = bt_deadline
    node_count[0] = 0
    backtrack(0, set(), [], 0)
    deadline = old_deadline
    
    # Phase 2: Local search with shuffling and exhaustive shrinking
    current = [r[:] for r in best_rows]
    current_scope = best_scope
    
    def try_local_search():
        nonlocal current, current_scope, best_rows, best_scope
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle row order to break symmetry
            for r_idx in range(n_rows):
                for m_idx in range(1, len(current[r_idx])):
                    orig = current[r_idx][m_idx]
                    # Try to shrink: move element down
                    for delta in range(-50, 0):
                        new_val = orig + delta
                        if new_val <= 0:
                            continue
                        if m_idx > 1 and new_val <= current[r_idx][m_idx-1]:
                            continue
                        if m_idx < len(current[r_idx])-1 and new_val >= current[r_idx][m_idx+1]:
                            continue
                        current[r_idx][m_idx] = new_val
                        if is_valid(current):
                            new_scope = get_scope(current)
                            if new_scope < current_scope:
                                current_scope = new_scope
                                improved = True
                                if current_scope < best_scope:
                                    best_scope = current_scope
                                    best_rows = [r[:] for r in current]
                            else:
                                current[r_idx][m_idx] = orig
                        else:
                            current[r_idx][m_idx] = orig
                    # Also try small expansions and contractions
                    for delta in range(1, 6):
                        new_val = orig + delta
                        if m_idx < len(current[r_idx])-1 and new_val >= current[r_idx][m_idx+1]:
                            continue
                        current[r_idx][m_idx] = new_val
                        if is_valid(current):
                            new_scope = get_scope(current)
                            if new_scope < current_scope:
                                current_scope = new_scope
                                improved = True
                                if current_scope < best_scope:
                                    best_scope = current_scope
                                    best_rows = [r[:] for r in current]
                            else:
                                current[r_idx][m_idx] = orig
                        else:
                            current[r_idx][m_idx] = orig
        
        # Random perturbation if stuck
        if time.time() < deadline:
            for _ in range(50):
                if time.time() >= deadline:
                    break
                r_idx = random.randint(0, n_rows-1)
                m_idx = random.randint(1, len(current[r_idx])-1)
                delta = random.randint(-15, 15)
                new_val = current[r_idx][m_idx] + delta
                if new_val > 0 and (m_idx == 1 or new_val > current[r_idx][m_idx-1]) and \
                   (m_idx == len(current[r_idx])-1 or new_val < current[r_idx][m_idx+1]):
                    current[r_idx][m_idx] = new_val
            if is_valid(current):
                new_scope = get_scope(current)
                if new_scope < current_scope:
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current]
            else:
                current = [r[:] for r in best_rows]
                current_scope = best_scope
    
    # Run local search
    try_local_search()
    
    # Phase 3: More aggressive randomized construction with pruning
    while time.time() < deadline:
        rows, scope = try_construct_randomized(n_rows, k, best_scope)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            if best_scope <= 105:
                break
        # If we're close to the bound, do another local search pass
        if best_scope <= 120 and time.time() < deadline:
            current = [r[:] for r in best_rows]
            current_scope = best_scope
            try_local_search()
    
    return best_rows

def try_construct_randomized(n_rows, k, current_best_scope):
    used = set()
    rows = []
    
    for i in range(n_rows):
        row = [0]
        last = 0
        
        while len(row) < k + 1:
            candidates = []
            window_size = 30
            x = last
            search_limit = last + 100
            
            while x < search_limit and len(candidates) < 10:
                x += 1
                if x >= current_best_scope:
                    return None, float('inf')
                
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    candidates.append((x, new_diffs))
            
            if not candidates:
                return None, float('inf')
            
            if random.random() < 0.3:
                x, new_diffs = candidates[0]
            else:
                pool = candidates[:5]
                x, new_diffs = random.choice(pool)
            
            row.append(x)
            used.update(new_diffs)
            last = x
        
        rows.append(row)
    
    scope = max(max(r) for r in rows)
    return rows, scope
