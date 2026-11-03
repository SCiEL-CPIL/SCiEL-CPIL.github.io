import time
import random
import math
from collections import Counter

def _scope(rows):
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

def _fast_greedy(k, n_rows, rng, spread=False):
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
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _build_all_diffs(rows):
    all_diffs = []
    for r in rows:
        all_diffs.extend(_diffs_for_row(r))
    return all_diffs

def _backtracking_solve(k, n_rows, max_scope, used_diffs, row_idx, current_row, deadline, rng):
    """
    Backtracking solver that tries to complete rows given used_diffs and max_scope.
    Returns True if a valid completion exists, and fills current_row.
    """
    if time.time() > deadline:
        return False
    
    if row_idx == n_rows:
        return True
    
    # Try to build row row_idx
    def build_row(idx, row, used_local, start_val):
        if time.time() > deadline:
            return False
        
        if len(row) == k + 1:
            # All entries chosen, check if this row is valid and record it
            return True
        
        # Determine search space for next entry
        min_next = row[-1] + 1
        max_next = max_scope
        
        # Pruning: if we can't fit remaining entries within max_scope
        remaining_entries = k + 1 - len(row)
        if min_next + (remaining_entries - 1) > max_scope:
            return False
        
        # Try values in random order for diversity
        candidates = list(range(min_next, max_next + 1))
        rng.shuffle(candidates)
        
        for val in candidates:
            # Check if val creates any conflict
            conflict = False
            new_diffs = []
            for a in row:
                d = val - a
                if d in used_local:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if conflict:
                continue
            
            # Tentatively add
            old_used = set(used_local)
            used_local.update(new_diffs)
            row.append(val)
            
            if build_row(idx, row, used_local, val):
                return True
            
            # Backtrack
            row.pop()
            used_local = old_used
        
        return False
    
    row = [0]
    if build_row(row_idx, row, used_diffs, 0):
        current_row[row_idx] = row
        return True
    
    return False

def _solve_with_backtracking(k, n_rows, max_scope, deadline, rng):
    """
    Try to find a valid set of rows with all entries <= max_scope.
    Uses backtracking with randomized ordering.
    """
    used_diffs = set()
    rows = [None] * n_rows
    
    def solve_rows(idx):
        if time.time() > deadline:
            return False
        
        if idx == n_rows:
            return True
        
        # Try to build row idx
        row = [0]
        
        def build_row():
            if time.time() > deadline:
                return False
            
            if len(row) == k + 1:
                return True
            
            min_next = row[-1] + 1
            max_next = max_scope
            
            # Pruning
            remaining = k + 1 - len(row)
            if min_next + (remaining - 1) > max_scope:
                return False
            
            # Generate candidates with some heuristics
            candidates = []
            for val in range(min_next, max_next + 1):
                # Quick check: does val create any difference already in used_diffs?
                ok = True
                for a in row:
                    if (val - a) in used_diffs:
                        ok = False
                        break
                if ok:
                    candidates.append(val)
            
            if not candidates:
                return False
            
            # Shuffle for diversity
            rng.shuffle(candidates)
            
            for val in candidates:
                new_diffs = set()
                for a in row:
                    new_diffs.add(val - a)
                
                # Check no internal conflicts
                if len(new_diffs) != len([val - a for a in row]):
                    continue
                
                # Tentatively use these diffs
                old_used = set(used_diffs)
                used_diffs.update(new_diffs)
                row.append(val)
                
                if build_row():
                    return True
                
                # Backtrack
                row.pop()
                used_diffs = old_used
            
            return False
        
        if build_row():
            rows[idx] = list(row)
            return True
        
        return False
    
    if solve_rows(0):
        return rows
    return None

def _sa_optimize(rows, k, deadline, rng, max_iters=50000):
    """
    Simulated Annealing to minimize the scope.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    # Track all differences for quick collision checking
    all_diffs = _build_all_diffs(current)
    diff_count = Counter(all_diffs)
    collisions = sum(max(0, c - 1) for c in diff_count.values())
    
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    
    T = 50.0
    cooling = 0.999
    min_T = 0.05
    
    # Track which entries are at the current scope (candidates for reduction)
    scope_entries = set()
    for i, r in enumerate(current):
        for j, v in enumerate(r):
            if v == current_scope:
                scope_entries.add((i, j))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Choose a random row and position to modify
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 100
        
        # Determine move type
        move_type = rng.random()
        
        if move_type < 0.3 and (i, j) in scope_entries:
            # Scope compression: try to reduce a scope entry
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val - 1))
            if new_val >= old_val:
                new_val = old_val - 1
            if new_val <= prev_val:
                continue
        elif move_type < 0.7:
            # Small perturbation
            delta = rng.randint(-3, 3)
            new_val = old_val + delta
        else:
            # Larger perturbation
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 15))
        
        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        
        # Compute the difference changes
        old_diffs = _diffs_for_row(current[i])
        
        # Update the row temporarily
        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)
        
        # Check validity of new row
        valid_row = len(set(new_diffs)) == len(new_diffs) and all(d > 0 for d in new_diffs)
        
        if not valid_row:
            current[i][j] = old_val
            continue
        
        # Compute scope change
        new_scope = max(_scope(current), max(new_row))
        
        # Compute collision change
        # Remove old diffs from count
        for d in old_diffs:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        
        # Add new diffs
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        
        new_collisions = sum(max(0, c - 1) for c in diff_count.values())
        
        # Cost function
        scope_penalty = 5000 if current_scope > 120 else 3000 if current_scope > 110 else 2000
        old_cost = current_scope + scope_penalty * collisions
        new_cost = new_scope + scope_penalty * new_collisions
        
        # Accept if better or with probability based on temperature
        if new_cost < old_cost:
            accept = True
        else:
            delta_cost = new_cost - old_cost
            if delta_cost < 0:
                accept = True
            else:
                accept_prob = math.exp(-delta_cost / max(T, min_T))
                accept = rng.random() < accept_prob
        
        if accept:
            # Update scope entries
            if old_val == current_scope:
                scope_entries.discard((i, j))
            if new_val == new_scope:
                scope_entries.add((i, j))
            for idx, r in enumerate(current):
                for jj, v in enumerate(r):
                    if v == new_scope and v != current_scope:
                        scope_entries.add((idx, jj))
                    elif v == current_scope and v != new_scope:
                        scope_entries.discard((idx, jj))
            
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            current[i][j] = old_val
            # Restore diff counts
            for d in new_diffs:
                diff_count[d] -= 1
                if diff_count[d] == 0:
                    del diff_count[d]
            for d in old_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
        
        T = max(T * cooling, min_T)
    
    return best_scope, best

def construct(n_rows, k):
    deadline = time.time() + 140.0  # Leave 10s margin
    rng = random.Random(123)
    
    # Phase 1: Generate initial candidates with greedy
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    for seed in range(3):
        r = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                max_step = r.randint(1, 10)
                found = False
                for _ in range(max_step):
                    x += r.randint(1, 3)
                    conflict = False
                    for a in row:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        found = True
                        break
                if not found:
                    while True:
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
                            break
            rows.append(row)
        candidates.append(rows)
    
    candidates.sort(key=_scope)
    
    # Phase 2: Use SA to improve the best candidate
    best_result = None
    best_score = float('inf')
    
    if time.time() < deadline - 60:
        sa_deadline = time.time() + min(40, deadline - time.time() - 60)
        s, rows = _sa_optimize(candidates[0], k, sa_deadline, rng, max_iters=20000)
        if _valid(rows, k):
            best_score = s
            best_result = rows
    
    # Phase 3: Use backtracking to prove existence at lower scopes
    # Binary search on scope
    if best_result is not None and time.time() < deadline - 30:
        lo = best_score - 10  # Try to find something better
        hi = best_score
        
        # First, try a few specific values below current best
        test_values = []
        for v in range(max(105, lo), hi + 1):
            test_values.append(v)
        
        # Shuffle test values for diversity
        rng.shuffle(test_values)
        
        bt_deadline = deadline - 10
        for scope_val in test_values:
            if time.time() > bt_deadline:
                break
            
            # Try backtracking with this scope
            result = _solve_with_backtracking(k, n_rows, scope_val, bt_deadline, rng)
            if result is not None and _valid(result, k):
                s = _scope(result)
                if s < best_score:
                    best_score = s
                    best_result = result
                # If we found a valid solution, we can try even lower
                break
    
    # Phase 4: If time remains, do more SA on the best result
    if best_result is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 5:
            s, rows = _sa_optimize(best_result, k, deadline - 3, rng, max_iters=int(remaining * 20000))
            if _valid(rows, k) and s < best_score:
                best_score = s
                best_result = rows
    
    if best_result is not None:
        return best_result
    
    return candidates[0]
