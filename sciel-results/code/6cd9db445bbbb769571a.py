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

def _build_diff_counter(rows):
    c = Counter()
    for r in rows:
        for d in _diffs_for_row(r):
            c[d] += 1
    return c

def _collision_count_from_counter(counter):
    return sum(max(0, v - 1) for v in counter.values())

def _fast_greedy(n_rows, k, rng, scope_limit=None):
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
    n_rows = len(rows)
    if max_iters is None:
        max_iters = 2000000
    
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    diff_counter = _build_diff_counter(current)
    collisions = _collision_count_from_counter(diff_counter)
    
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    
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
        
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 10000
        
        move_type = rng.random()
        
        if move_type < 0.5:
            upper = old_val - 1 if old_val > prev_val + 1 else prev_val
            if upper <= prev_val:
                continue
            new_val = rng.randint(prev_val + 1, upper)
        elif move_type < 0.8:
            delta = rng.randint(-8, 8)
            new_val = old_val + delta
        else:
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 30))
        
        if new_val <= prev_val or (j < k and new_val >= next_val):
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
        
        new_collisions = _collision_count_from_counter(diff_counter)
        
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
        
        T = max(T * cooling, min_T)
    
    return best_scope, best

def _construct_greedy_random(n_rows, k, rng, scope_limit):
    """
    Construct rows using random greedy with restarts.
    Each row is built by picking the next smallest valid value,
    but with some randomness to escape local optima.
    """
    for restart in range(200):
        used = set()
        rows = []
        success = True
        
        for row_idx in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if x > scope_limit:
                    success = False
                    break
                # Check if x is valid
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    # Add with some probability to introduce randomness
                    # But for the first few rows, be deterministic
                    if row_idx < 3 or rng.random() < 0.8:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                    else:
                        # Skip and try next value
                        continue
            if not success:
                break
            rows.append(row)
        
        if success and len(rows) == n_rows:
            s = _scope(rows)
            if s <= scope_limit:
                return rows
    
    return None

def _pairwise_swap_search(rows, k, deadline, rng, max_iters=50000):
    """
    Local search that swaps pairs of values between rows.
    This is a structural change from single-value SA.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    diff_counter = _build_diff_counter(current)
    collisions = _collision_count_from_counter(diff_counter)
    
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    
    collision_weight = 100000
    
    def cost(scope, coll):
        return coll * collision_weight + scope
    
    it = 0
    no_improve = 0
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1
        
        # Choose two different rows
        if n_rows < 2:
            break
        i1, i2 = rng.sample(range(n_rows), 2)
        
        # Choose positions in each row
        j1 = rng.randint(1, k)
        j2 = rng.randint(1, k)
        
        v1 = current[i1][j1]
        v2 = current[i2][j2]
        
        # Try to swap or adjust
        if v1 == v2:
            continue
        
        # Compute what would happen if we swap these values
        # But we need to maintain strictly increasing order within each row
        
        # For row i1: replacing v1 with v2
        # Check if v2 fits in row i1 at position j1
        prev1 = current[i1][j1-1]
        next1 = current[i1][j1+1] if j1 < k else v1 + 10000
        if v2 <= prev1 or (j1 < k and v2 >= next1):
            # Try adjusting v2 to fit
            if v2 <= prev1:
                v2_new = prev1 + 1
            else:
                v2_new = min(v2, next1 - 1) if j1 < k else v2
            if v2_new <= prev1 or (j1 < k and v2_new >= next1):
                continue
        else:
            v2_new = v2
        
        # For row i2: replacing v2 with v1
        prev2 = current[i2][j2-1]
        next2 = current[i2][j2+1] if j2 < k else v2 + 10000
        if v1 <= prev2 or (j2 < k and v1 >= next2):
            if v1 <= prev2:
                v1_new = prev2 + 1
            else:
                v1_new = min(v1, next2 - 1) if j2 < k else v1
            if v1_new <= prev2 or (j2 < k and v1_new >= next2):
                continue
        else:
            v1_new = v1
        
        # Apply changes
        old_row1 = current[i1][:]
        old_row2 = current[i2][:]
        old_diffs1 = _diffs_for_row(old_row1)
        old_diffs2 = _diffs_for_row(old_row2)
        
        current[i1][j1] = v2_new
        current[i2][j2] = v1_new
        
        new_row1 = current[i1]
        new_row2 = current[i2]
        new_diffs1 = _diffs_for_row(new_row1)
        new_diffs2 = _diffs_for_row(new_row2)
        
        # Check internal validity
        if len(new_diffs1) != len(set(new_diffs1)) or len(new_diffs2) != len(set(new_diffs2)):
            current[i1] = old_row1
            current[i2] = old_row2
            continue
        
        # Update diff counter
        for d in old_diffs1:
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        for d in old_diffs2:
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        
        for d in new_diffs1:
            diff_counter[d] = diff_counter.get(d, 0) + 1
        for d in new_diffs2:
            diff_counter[d] = diff_counter.get(d, 0) + 1
        
        new_collisions = _collision_count_from_counter(diff_counter)
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
                no_improve = 0
            else:
                no_improve += 1
        else:
            current[i1] = old_row1
            current[i2] = old_row2
            for d in new_diffs1:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in new_diffs2:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in old_diffs1:
                diff_counter[d] = diff_counter.get(d, 0) + 1
            for d in old_diffs2:
                diff_counter[d] = diff_counter.get(d, 0) + 1
        
        T = max(T * cooling, min_T)
        
        if no_improve > 10000:
            no_improve = 0
            T = min(T * 2, 50.0)
    
    return best_scope, best

def _row_rebuild_search(rows, k, deadline, rng):
    """
    Try to rebuild individual rows to reduce scope.
    Pick a row, remove it, and rebuild it with a tighter scope constraint.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    best = [list(r) for r in current]
    best_scope = current_scope
    
    # Precompute diffs for all rows except one
    def compute_used_diffs_except(exclude_idx):
        used = set()
        for i, r in enumerate(current):
            if i == exclude_idx:
                continue
            for d in _diffs_for_row(r):
                used.add(d)
        return used
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Pick a row to rebuild
        row_idx = rng.randint(0, n_rows - 1)
        old_row = current[row_idx]
        old_max = old_row[-1]
        
        used = compute_used_diffs_except(row_idx)
        
        # Try to rebuild this row with a smaller max
        new_max_target = old_max - rng.randint(1, 5)
        if new_max_target < k:  # Minimum possible
            new_max_target = k
        
        # Backtracking to build a new row
        new_row = [0]
        new_diffs = set()
        
        def backtrack(pos, max_val):
            if time.time() > deadline:
                return False
            if pos == k + 1:
                return True
            
            prev = new_row[-1]
            min_next = prev + 1
            
            for val in range(min_next, max_val + 1):
                conflict = False
                tmp_diffs = []
                for idx in range(pos):
                    d = val - new_row[idx]
                    if d in used or d in new_diffs:
                        conflict = True
                        break
                    tmp_diffs.append(d)
                
                if conflict:
                    continue
                
                new_row.append(val)
                for d in tmp_diffs:
                    new_diffs.add(d)
                
                if backtrack(pos + 1, max_val):
                    return True
                
                new_row.pop()
                for d in tmp_diffs:
                    new_diffs.discard(d)
            
            return False
        
        if backtrack(1, new_max_target):
            # Check if this reduces scope
            new_row_max = new_row[-1]
            new_global_scope = max(new_row_max, max(r[-1] for i, r in enumerate(current) if i != row_idx))
            
            if new_global_scope < current_scope:
                current[row_idx] = new_row
                current_scope = new_global_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best = [list(r) for r in current]
        
        # If we didn't improve, try with slightly larger target
        if time.time() > deadline:
            break
    
    return best_scope, best

def construct(n_rows, k):
    deadline = time.time() + 140.0
    rng = random.Random(12345)
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Try to construct with target scopes from low to high
    target_scopes = [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 120, 125, 130]
    
    for target in target_scopes:
        if time.time() > deadline - 40:
            break
        
        # Try random greedy construction
        t0 = time.time()
        rows = _construct_greedy_random(n_rows, k, rng, target)
        t1 = time.time()
        
        if rows is not None:
            s = _scope(rows)
            if s < best_score:
                best_score = s
                best_result = rows
            if s <= target:
                break
    
    # Fallback to simple greedy if nothing found
    if best_result is None:
        rows = _fast_greedy(n_rows, k, rng)
        if rows is not None:
            best_result = rows
            best_score = _scope(rows)
        else:
            # Last resort
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
    
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 30:
        remaining = deadline - time.time()
        sa_deadline = time.time() + remaining * 0.4
        max_iters = int(remaining * 40000)
        s, rows = _sa_optimize(best_result, k, sa_deadline, rng, max_iters=max_iters)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
    
    # Phase 3: Pairwise swap search
    if time.time() < deadline - 20:
        remaining = deadline - time.time()
        swap_deadline = time.time() + remaining * 0.4
        s, rows = _pairwise_swap_search(best_result, k, swap_deadline, rng, max_iters=int(remaining * 20000))
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
    
    # Phase 4: Row rebuild search
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        rebuild_deadline = time.time() + remaining * 0.8
        s, rows = _row_rebuild_search(best_result, k, rebuild_deadline, rng)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
    
    # Final validation
    if best_result is not None and not _valid(best_result, k):
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
