import time
import random
import math
from collections import defaultdict

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

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    s = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            s.add(row[j] - row[i])
    return s

def _build_diff_counter(rows):
    counter = defaultdict(int)
    for r in rows:
        for d in _diffs_for_row(r):
            counter[d] += 1
    return counter

def _collision_count(counter):
    return sum(max(0, v - 1) for v in counter.values())

def _construct_rows_backtrack(n_rows, k, deadline, rng, target_scope):
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        current_row = [0]
        current_diffs = set()
        
        def backtrack(pos, current_row, current_diffs, used_diffs, k, max_val, deadline, rng):
            if time.time() > deadline:
                return None
            
            n = k + 1
            if pos == n:
                return list(current_row)
            
            prev = current_row[-1]
            min_next = prev + 1
            
            if min_next + (n - pos - 1) > max_val:
                return None
            
            candidates = list(range(min_next, max_val + 1))
            
            if len(candidates) > 10:
                head = candidates[:5]
                tail = candidates[5:]
                rng.shuffle(tail)
                candidates = head + tail
            
            for val in candidates:
                if time.time() > deadline:
                    return None
                
                conflict = False
                new_diffs = []
                for idx in range(pos):
                    d = val - current_row[idx]
                    if d in current_diffs or d in used_diffs:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if conflict:
                    continue
                
                current_row.append(val)
                current_diffs.update(new_diffs)
                
                result = backtrack(pos + 1, current_row, current_diffs, used_diffs, k, max_val, deadline, rng)
                
                current_row.pop()
                for d in new_diffs:
                    current_diffs.discard(d)
                
                if result is not None:
                    return result
            
            return None

        row = backtrack(1, current_row, current_diffs, used_diffs, k, target_scope, deadline, rng)
        
        if row is None:
            return None
        
        rows.append(row)
        for d in _diffs_for_row(row):
            used_diffs.add(d)
    
    return rows

def _greedy_construct(n_rows, k, rng):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            conflict = False
            for a in row:
                if (x - a) in used:
                    conflict = True
                    break
            if not conflict:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows

def _sa_optimize(rows, n_rows, k, deadline, rng):
    """
    Fast simulated annealing with delta-based conflict tracking.
    Returns (best_rows, best_scope).
    """
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    best_rows = [list(r) for r in current]
    best_scope = current_scope
    
    # Build diff counter
    diff_counter = _build_diff_counter(current)
    collisions = _collision_count(diff_counter)
    
    T = 10.0
    cooling = 0.995
    min_T = 0.001
    
    # Precompute row diffs for quick update
    row_diffs = [_diffs_for_row(r) for r in current]
    
    while time.time() < deadline:
        # Pick random row and position
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 100000
        
        # Generate new value with bias toward reduction
        if rng.random() < 0.6:
            # Try to reduce
            upper = old_val - 1
            if upper <= prev_val:
                continue
            new_val = rng.randint(prev_val + 1, upper)
        else:
            # Small perturbation
            delta = rng.randint(-8, 8)
            new_val = old_val + delta
        
        if new_val <= prev_val or (j < k and new_val >= next_val) or new_val == old_val:
            continue
        
        # Compute new row and its diffs
        new_row = current[i][:]
        new_row[j] = new_val
        new_diffs = _diffs_for_row(new_row)
        
        # Check internal row validity
        if len(new_diffs) != len(set(new_diffs)):
            continue
        
        # Delta update: compute collision change
        old_diffs = row_diffs[i]
        
        # Count how many old diffs become unique (collision decreases)
        # and how many new diffs create collisions (collision increases)
        delta_collisions = 0
        
        # For each old diff: if it was the only occurrence, removing it doesn't change collision
        # if it had count > 1, removing one decreases collision by 1
        for d in old_diffs:
            c = diff_counter[d]
            if c > 1:
                # After removal, count becomes c-1. 
                # Collision contribution: max(0, c-1) - max(0, c-1) ... wait
                # Old contribution: max(0, c-1)
                # New contribution after removal: max(0, (c-1)-1) = max(0, c-2)
                # Delta = max(0, c-2) - max(0, c-1)
                old_contrib = max(0, c - 1)
                new_contrib = max(0, c - 2)
                delta_collisions += new_contrib - old_contrib
        
        # For each new diff: adding one
        for d in new_diffs:
            c = diff_counter[d]
            # Old contribution: max(0, c-1)
            # New contribution after adding: max(0, (c+1)-1) = max(0, c)
            old_contrib = max(0, c - 1)
            new_contrib = max(0, c)
            delta_collisions += new_contrib - old_contrib
        
        new_collisions = collisions + delta_collisions
        
        # Compute new scope
        new_scope = current_scope
        if new_val > new_scope and j == k:
            new_scope = new_val
        elif new_scope == old_val:
            # Need to recompute if the max was at this position
            # Find next max
            new_scope = max(a for r in current for a in r[1:])
            if new_val > new_scope:
                new_scope = new_val
        
        # Cost function
        old_cost = current_scope + collisions * 100
        new_cost = new_scope + new_collisions * 100
        
        accept = False
        if new_cost <= old_cost:
            accept = True
        else:
            if T > min_T:
                prob = math.exp(-(new_cost - old_cost) / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            # Commit changes
            for d in old_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in new_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
            
            current[i] = new_row
            row_diffs[i] = new_diffs
            current_scope = new_scope
            collisions = new_collisions
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current]
        else:
            # No change needed since we didn't modify anything
            pass
        
        T = max(T * cooling, min_T)
    
    return best_rows, best_scope

def _local_search_polish(rows, n_rows, k, deadline, rng):
    """
    Final polish: try to reduce scope by moving entries down.
    """
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    best_rows = [list(r) for r in current]
    best_scope = current_scope
    
    diff_counter = _build_diff_counter(current)
    row_diffs = [_diffs_for_row(r) for r in current]
    
    # Try to reduce each entry to the minimum valid value
    for _ in range(10000):
        if time.time() > deadline:
            break
        
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 100000
        
        # Try to set to minimum possible
        min_val = prev_val + 1
        if j < k:
            min_val = max(min_val, next_val - 1)
        
        if min_val >= old_val:
            continue
        
        # Try values from min_val upward
        found = False
        for new_val in range(min_val, old_val):
            new_row = current[i][:]
            new_row[j] = new_val
            new_diffs = _diffs_for_row(new_row)
            
            if len(new_diffs) != len(set(new_diffs)):
                continue
            
            # Check global validity
            old_diffs = row_diffs[i]
            
            # Temporarily remove old diffs
            temp_counter = dict(diff_counter)
            for d in old_diffs:
                temp_counter[d] -= 1
                if temp_counter[d] == 0:
                    del temp_counter[d]
            
            # Check new diffs against temp_counter
            valid = True
            for d in new_diffs:
                if temp_counter.get(d, 0) > 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Accept this reduction
            for d in old_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in new_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
            
            current[i] = new_row
            row_diffs[i] = new_diffs
            found = True
            break
        
        if found:
            new_scope = _scope(current)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current]
    
    return best_rows, best_scope

def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin
    rng = random.Random(42)
    
    best_result = None
    best_scope = float('inf')
    
    min_possible_scope = n_rows * k * (k + 1) // 2
    current_target = min_possible_scope + 2
    
    while time.time() < deadline - 30:
        if current_target > 200:
            break
        
        bt_deadline = time.time() + 5.0
        rows = _construct_rows_backtrack(n_rows, k, bt_deadline, rng, current_target)
        
        if rows is not None:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_result = rows
            break
        
        current_target += 1
    
    if best_result is None:
        best_result = _greedy_construct(n_rows, k, rng)
        best_scope = _scope(best_result)
    
    # SA optimization
    if time.time() < deadline - 15:
        sa_deadline = time.time() + (deadline - time.time()) * 0.7
        best_result, best_scope = _sa_optimize(best_result, n_rows, k, sa_deadline, rng)
    
    # Final polish
    if time.time() < deadline - 2:
        polish_deadline = time.time() + 2.0
        best_result, best_scope = _local_search_polish(best_result, n_rows, k, polish_deadline, rng)
    
    return best_result
