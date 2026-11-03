import time
import random
import math

def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _row_diffs(row):
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k, rng, max_scope=150):
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        
        def backtrack(pos, current_row, current_diffs):
            if pos == k + 1:
                return True
            
            prev = current_row[-1]
            min_next = prev + 1
            
            # Try values in random order, biased to smaller
            # Limit search space to max_scope
            candidates = list(range(min_next, max_scope + 1))
            # Shuffle to avoid getting stuck in the same local optimum
            rng.shuffle(candidates)
            
            for val in candidates:
                # Check internal conflicts
                conflict = False
                new_diffs = []
                for i in range(pos):
                    d = val - current_row[i]
                    if d in current_diffs:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if conflict:
                    continue
                
                # Check global conflicts
                if any(d in used_diffs for d in new_diffs):
                    continue
                
                current_row.append(val)
                current_diffs.update(new_diffs)
                
                if backtrack(pos + 1, current_row, current_diffs):
                    return True
                
                current_row.pop()
                for d in new_diffs:
                    current_diffs.discard(d)
            
            return False
        
        success = False
        # Try increasing max_scope if needed
        for scope_limit in [120, 130, 140, 150, 200]:
            current_row = [0]
            current_diffs = set()
            if backtrack(1, current_row, current_diffs):
                rows.append(current_row)
                for d in current_diffs:
                    used_diffs.add(d)
                success = True
                break
        
        if not success:
            return None
            
    return rows

def _sa_optimize(rows, k, deadline, rng):
    n_rows = len(rows)
    current = [list(r) for r in rows]
    
    # Determine max scope for array
    MAX_SCOPE = 200
    
    global_counts = [0] * (MAX_SCOPE + 1)
    current_scope = 0
    
    for r in current:
        diffs = _row_diffs(r)
        for d in diffs:
            if d > MAX_SCOPE:
                # Extend if necessary, though unlikely
                pass
            global_counts[d] += 1
            if d > current_scope:
                current_scope = d
    
    def get_collisions():
        c = 0
        for count in global_counts:
            if count > 1:
                c += count - 1
        return c
    
    current_collisions = get_collisions()
    
    best = [list(r) for r in current]
    best_scope = current_scope if current_collisions == 0 else float('inf')
    best_cost = current_collisions * 10000 + current_scope
    
    T = 50.0
    cooling = 0.9999
    min_T = 0.01
    
    start_time = time.time()
    
    while time.time() - start_time < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 1000
        
        # Move
        if old_val == current_scope:
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val - 1))
            if new_val >= old_val:
                new_val = old_val - 1
        else:
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
        
        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        
        # Store old diffs
        old_row = list(current[i])
        old_diffs = _row_diffs(old_row)
        
        # Apply move
        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _row_diffs(new_row)
        
        # Check internal validity
        if len(new_diffs) != len(set(new_diffs)):
            current[i][j] = old_val
            continue
        
        # Update counts
        for d in old_diffs:
            global_counts[d] -= 1
        for d in new_diffs:
            if d > MAX_SCOPE:
                # Handle out of bounds
                pass
            else:
                global_counts[d] += 1
        
        # Recalculate scope and collisions
        new_scope = current_scope
        if old_val == current_scope and new_val < old_val:
            # Recalculate max
            new_scope = 0
            for r in current:
                if r[-1] > new_scope:
                    new_scope = r[-1]
        elif new_val > new_scope:
            new_scope = new_val
            
        new_collisions = get_collisions()
        
        new_cost = new_collisions * 10000 + new_scope
        
        # Acceptance
        if new_cost <= best_cost:
            accept = True
        else:
            delta_cost = new_cost - (current_collisions * 10000 + current_scope)
            # We need the current cost to compare for delta
            current_cost = current_collisions * 10000 + current_scope
            delta_cost = new_cost - current_cost
            accept_prob = math.exp(-delta_cost / max(T, min_T))
            accept = rng.random() < accept_prob
        
        if accept:
            current_scope = new_scope
            current_collisions = new_collisions
            
            if current_collisions == 0 and current_scope < best_scope:
                best_scope = current_scope
                best = [list(r) for r in current]
                best_cost = current_scope
        else:
            # Revert
            current[i][j] = old_val
            for d in new_diffs:
                if d <= MAX_SCOPE:
                    global_counts[d] -= 1
            for d in old_diffs:
                if d <= MAX_SCOPE:
                    global_counts[d] += 1
        
        T = max(T * cooling, min_T)
        
        # Periodic restart? No, just keep going.
        
    return best_scope, best

def construct(n_rows, k):
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    rows = _greedy_construct(n_rows, k, rng)
    
    if rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if not any((x - a) in used for a in row):
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
    
    sa_deadline = deadline - 10.0
    best_scope, best_rows = _sa_optimize(rows, k, sa_deadline, rng)
    
    # Final check
    if not _valid(best_rows, k):
        # If SA failed to find a valid one (unlikely if init is valid), return init
        # But init is valid.
        # If best_rows is invalid, it means we never found a valid state in SA?
        # No, best is initialized to current (which is valid from greedy).
        # So best_rows should be valid.
        
        # Just in case
        if _valid(rows, k):
            return rows
        else:
            return best_rows
            
    return best_rows
