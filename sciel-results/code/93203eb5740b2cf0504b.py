import time
import random
import math
from collections import Counter

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

def _all_diffs(rows):
    """Return a list of all differences across all rows."""
    all_d = []
    for r in rows:
        all_d.extend(_diffs_for_row(r))
    return all_d

def _collision_count(all_diffs):
    """Count the number of collisions (duplicates) in the diff list."""
    c = Counter(all_diffs)
    return sum(max(0, v - 1) for v in c.values())

def _build_diff_counter(rows):
    """Build a Counter of all differences."""
    c = Counter()
    for r in rows:
        for d in _diffs_for_row(r):
            c[d] += 1
    return c

def _fast_greedy(n_rows, k, rng, scope_limit=None):
    """
    Fast greedy construction. For each row, greedily pick the smallest
    value that doesn't create a duplicate difference.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None
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

def _sa_optimize(rows, k, deadline, rng, max_iters=None):
    """
    Simulated Annealing to minimize scope with collision penalty.
    Includes single-element moves and swap moves between rows.
    """
    n_rows = len(rows)
    if max_iters is None:
        max_iters = 2000000
    
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    # Build diff counter
    diff_counter = _build_diff_counter(current)
    collisions = _collision_count(list(diff_counter.elements()))
    
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    
    # Cost function
    collision_weight = 100000
    
    def cost(scope, coll):
        return coll * collision_weight + scope
    
    T = 50.0
    cooling = 0.9999
    min_T = 0.01
    
    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1
        
        move_type = rng.random()
        
        if move_type < 0.6:
            # Single element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current[i][j]
            prev_val = current[i][j-1]
            next_val = current[i][j+1] if j < k else float('inf')
            
            if rng.random() < 0.5:
                upper = old_val - 1 if old_val > prev_val + 1 else prev_val
                if upper <= prev_val:
                    continue
                new_val = rng.randint(prev_val + 1, upper)
            else:
                delta = rng.randint(-8, 8)
                new_val = old_val + delta
            
            if new_val <= prev_val or new_val >= next_val:
                continue
            if new_val == old_val:
                continue
            
            old_row = current[i]
            old_diffs = _diffs_for_row(old_row)
            
            current[i][j] = new_val
            new_row = current[i]
            new_diffs = _diffs_for_row(new_row)
            
            if len(new_diffs) != len(set(new_diffs)):
                current[i][j] = old_val
                continue
            
            for d in old_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            for d in new_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
            
            new_collisions = _collision_count(list(diff_counter.elements()))
            
            new_scope = current_scope
            if new_val > current_scope:
                new_scope = new_val
            elif old_val == current_scope and new_val < old_val:
                new_scope = _scope(current)
            
            old_c = cost(current_scope, collisions)
            new_c = cost(new_scope, new_collisions)
            
            if new_c <= old_c:
                accept = True
            else:
                delta_c = new_c - old_c
                prob = math.exp(-delta_c / max(T, min_T))
                accept = rng.random() < prob
            
            if accept:
                current_scope = new_scope
                collisions = new_collisions
                if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                    best_scope = new_scope
                    best_collisions = new_collisions
                    best = [list(r) for r in current]
            else:
                current[i][j] = old_val
                for d in new_diffs:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for d in old_diffs:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                    
        elif move_type < 0.9 and n_rows >= 2:
            # Swap move
            i, j = rng.sample(range(n_rows), 2)
            p = rng.randint(1, k)
            q = rng.randint(1, k)
            
            val_i = current[i][p]
            val_j = current[j][q]
            
            if val_i == val_j:
                continue
            
            prev_i = current[i][p-1]
            next_i = current[i][p+1] if p < k else float('inf')
            if val_j <= prev_i or val_j >= next_i:
                continue
                
            prev_j = current[j][q-1]
            next_j = current[j][q+1] if q < k else float('inf')
            if val_i <= prev_j or val_i >= next_j:
                continue
            
            old_row_i = list(current[i])
            old_row_j = list(current[j])
            old_diffs_i = _diffs_for_row(old_row_i)
            old_diffs_j = _diffs_for_row(old_row_j)
            
            current[i][p] = val_j
            current[j][q] = val_i
            
            new_row_i = current[i]
            new_row_j = current[j]
            new_diffs_i = _diffs_for_row(new_row_i)
            new_diffs_j = _diffs_for_row(new_row_j)
            
            if len(new_diffs_i) != len(set(new_diffs_i)) or len(new_diffs_j) != len(set(new_diffs_j)):
                current[i][p] = val_i
                current[j][q] = val_j
                continue
            
            for d in old_diffs_i:
                diff_counter[d] -= 1
                if diff_counter[d] == 0: del diff_counter[d]
            for d in old_diffs_j:
                diff_counter[d] -= 1
                if diff_counter[d] == 0: del diff_counter[d]
                
            for d in new_diffs_i:
                diff_counter[d] = diff_counter.get(d, 0) + 1
            for d in new_diffs_j:
                diff_counter[d] = diff_counter.get(d, 0) + 1
                
            new_collisions = _collision_count(list(diff_counter.elements()))
            new_scope = _scope(current)
            
            old_c = cost(current_scope, collisions)
            new_c = cost(new_scope, new_collisions)
            
            if new_c <= old_c:
                accept = True
            else:
                delta_c = new_c - old_c
                prob = math.exp(-delta_c / max(T, min_T))
                accept = rng.random() < prob
            
            if accept:
                current_scope = new_scope
                collisions = new_collisions
                if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                    best_scope = new_scope
                    best_collisions = new_collisions
                    best = [list(r) for r in current]
            else:
                current[i][p] = val_i
                current[j][q] = val_j
                for d in new_diffs_i:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0: del diff_counter[d]
                for d in new_diffs_j:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0: del diff_counter[d]
                for d in old_diffs_i:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                for d in old_diffs_j:
                    diff_counter[d] = diff_counter.get(d, 0) + 1

        T = max(T * cooling, min_T)
    
    return best_scope, best

def _csp_backtrack_row(pos, current_row, current_diffs, used_diffs, k, max_val, deadline, rng):
    """
    Backtrack to build a single valid row of length k+1 starting at 0,
    avoiding conflicts with used_diffs and internal conflicts.
    Uses forward checking: at each step, verify that remaining positions
    can still be filled.
    """
    if time.time() > deadline:
        return None
    n = k + 1
    if pos == n:
        return list(current_row)
    
    prev = current_row[-1]
    min_next = prev + 1
    max_next = max(max_val, prev + (n - pos - 1))  # need at least this many more steps
    
    # Forward checking: ensure we can still place remaining elements
    # The smallest possible value for position pos is min_next
    # The smallest possible value for position pos+1 is min_next+1
    # etc. So the minimum possible max value is min_next + (n - pos - 1)
    # If this exceeds max_val, we're stuck
    if min_next + (n - pos - 1) > max_val:
        return None
    
    # Generate candidates
    candidates = list(range(min_next, max_next + 1))
    
    # Try smaller values first with some randomness in the upper portion
    if len(candidates) > 6:
        mid = len(candidates) // 2
        rng.shuffle(candidates[mid:])
    
    for val in candidates:
        if time.time() > deadline:
            return None
        
        # Check conflicts with current row's diffs and used_diffs
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
        
        result = _csp_backtrack_row(pos + 1, current_row, current_diffs, used_diffs, k, max_val, deadline, rng)
        
        current_row.pop()
        for d in new_diffs:
            current_diffs.discard(d)
        
        if result is not None:
            return result
    
    return None

def _construct_rows_csp(n_rows, k, deadline, rng, target_scope):
    """
    Construct n_rows valid rows using CSP backtracking with forward checking.
    """
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        best_row = None
        best_max_val = float('inf')
        
        # Try multiple attempts with different random orderings
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            current_row = [0]
            current_diffs = set()
            
            result = _csp_backtrack_row(1, current_row, current_diffs, used_diffs, k, target_scope, deadline, rng)
            
            if result is not None:
                row_max = result[-1]
                if row_max < best_max_val:
                    best_max_val = row_max
                    best_row = result
                if best_max_val <= target_scope:
                    break
        
        if best_row is None:
            return None
        
        rows.append(best_row)
        for d in _diffs_for_row(best_row):
            used_diffs.add(d)
    
    return rows

def _scope_squeeze(rows, k, deadline, rng):
    """
    Try to reduce the scope by 1 iteratively using local search.
    """
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        max_rows = [i for i, r in enumerate(rows) if r[-1] == current_scope]
        if not max_rows:
            break
        
        i = rng.choice(max_rows)
        row = rows[i]
        old_max = row[-1]
        prev = row[k-1]
        
        found = False
        for new_val in range(old_max - 1, prev, -1):
            row[k] = new_val
            new_diffs = _diffs_for_row(row)
            
            if len(new_diffs) != len(set(new_diffs)):
                continue
            
            # Check cross-row conflicts
            conflict = False
            for other_idx, other_row in enumerate(rows):
                if other_idx == i:
                    continue
                other_diffs_set = set(_diffs_for_row(other_row))
                for d in new_diffs:
                    if d in other_diffs_set:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                current_scope = new_val
                found = True
                break
        
        if not found:
            row[k] = old_max
            # Try a different row or give up
            continue
    
    return rows

def construct(n_rows, k):
    deadline = time.time() + 145.0
    rng = random.Random(12345)
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: CSP backtracking construction
    # Try to construct with target scope around the known best
    for target_scope in [112, 114, 116, 120, 125, 130]:
        if time.time() > deadline - 30:
            break
        
        rows = _construct_rows_csp(n_rows, k, deadline - 20, rng, target_scope)
        
        if rows is not None:
            s = _scope(rows)
            if s < best_score:
                best_score = s
                best_result = rows
            if s <= target_scope:
                break
    
    # Phase 2: Fallback to greedy if CSP failed
    if best_result is None:
        rows = _fast_greedy(n_rows, k, rng)
        if rows is not None:
            best_result = rows
            best_score = _scope(rows)
        else:
            # Last resort: simple sequential
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
            best_result = rows
            best_score = _scope(rows)
    
    # Phase 3: Simulated Annealing optimization
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        sa_deadline = time.time() + remaining * 0.75
        max_iters = int(remaining * 50000)
        s, rows = _sa_optimize(best_result, k, sa_deadline, rng, max_iters=max_iters)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
    
    # Phase 4: Scope squeeze
    if time.time() < deadline - 3:
        remaining = deadline - time.time()
        squeeze_deadline = time.time() + remaining * 0.8
        squeezed = _scope_squeeze(best_result, k, squeeze_deadline, rng)
        if _valid(squeezed, k):
            s = _scope(squeezed)
            if s < best_score:
                best_score = s
                best_result = squeezed
    
    # Final validation
    if best_result is not None and not _valid(best_result, k):
        # Shouldn't happen, but fallback
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
        best_result = rows
    
    return best_result
