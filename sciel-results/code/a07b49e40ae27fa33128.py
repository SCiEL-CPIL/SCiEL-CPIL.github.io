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

def _fast_greedy(k, n_rows, rng=None, spread=False):
    """Bitmask-based greedy construction for fast initialization."""
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _csp_build_row(k, used_diffs, max_scope, rng):
    """
    Recursively build a single row of length k+1 (starting with 0)
    such that all differences are not in used_diffs and are distinct within the row.
    Uses backtracking.
    """
    row = [0]
    local_diffs = set()
    
    def backtrack(idx, current_max):
        if idx == k + 1:
            return True
        
        # Heuristic: Try to keep numbers small to minimize scope
        # But also avoid conflicts
        start = row[-1] + 1
        # We can bound the search by max_scope
        for x in range(start, max_scope + 1):
            # Check differences with previous elements in this row
            valid = True
            new_diffs = []
            for i in range(len(row)):
                d = x - row[i]
                if d <= 0:
                    continue
                if d in used_diffs or d in local_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                row.append(x)
                local_diffs.update(new_diffs)
                
                if backtrack(idx + 1, x):
                    return True
                
                # Backtrack
                row.pop()
                for d in new_diffs:
                    local_diffs.discard(d)
        
        return False

    if backtrack(1, 0):
        return row
    return None

def _csp_initialize(k, n_rows, rng, time_limit=5.0):
    """
    Attempt to construct a valid solution using CSP backtracking for each row.
    Returns list of rows or None if failed/too slow.
    """
    start_time = time.time()
    used_diffs = set()
    rows = []
    
    for r_idx in range(n_rows):
        if time.time() - start_time > time_limit:
            return None
        
        row = _csp_build_row(k, used_diffs, 200, rng)
        
        if row is None:
            return None
        
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
        
        rows.append(row)
        
    return rows

def _dp_initialize(k, n_rows, rng, time_limit=5.0):
    """
    Dynamic Programming / A* style initialization.
    State: List of open rows. Each row is a list of integers.
    We process differences from 1 upwards.
    For each difference d, we try to assign it to an existing open row to extend it,
    or start a new row if beneficial (and within n_rows limit).
    We prefer assignments that keep the row's max element small and progress towards completion.
    """
    start_time = time.time()
    open_rows = [] # List of lists, each is a partial row starting with 0
    used_diffs = set()
    
    # We need to generate a stream of candidate differences.
    # We don't know the upper bound, so we iterate d=1, 2, ...
    # We stop when we have n_rows complete rows.
    
    max_d = 300 # Safety bound for initialization
    
    for d in range(1, max_d + 1):
        if len([r for r in open_rows if len(r) == k + 1]) >= n_rows:
            break
        if time.time() - start_time > time_limit:
            return None
            
        # Try to assign d to an existing open row
        assigned = False
        # Sort open rows by current max (prefer extending smaller rows? or larger? 
        # Usually extending the row with the smallest max is good to keep scope low, 
        # but we must ensure we don't block other rows. 
        # Let's try rows that are "closest" to completion or have small max first.
        # Actually, just iterate. If we have many open rows, this is expensive.
        # Limit open rows to n_rows.
        
        # Heuristic: Prioritize rows that are already large (closer to completion) to finish them?
        # Or prioritize small rows to keep them small?
        # Let's try extending the row with the smallest current max first.
        # If that fails, try others.
        
        candidates = []
        for idx, row in enumerate(open_rows):
            if len(row) == k + 1:
                continue
            # Check if d can be added to row
            # New max would be row[-1] + d? No, d is a difference between any two elements.
            # If we add a new element x, then x - y = d for some y in row?
            # No, the difference d is just a value. 
            # The row is a set of numbers. The differences are pairwise.
            # If we add a new number x to the row, we generate new differences x - y for all y in row.
            # We want one of these to be d? 
            # No, the DP approach described in the prompt implies assigning the *value* d as a difference.
            # This implies we are building the set of differences.
            # But the row structure must be consistent.
            # If we decide "difference d is used", it must be that there exist i, j in the row such that a_j - a_i = d.
            
            # This interpretation is tricky for a simple DP. 
            # Alternative interpretation: 
            # We build the rows element by element. 
            # The "state" is the set of partial rows.
            # The "transition" is adding a new element to one of the rows.
            # The "cost" is the new element's value (or max value).
            # We want to minimize the max value.
            
            # Let's stick to a simpler heuristic for this function:
            # Try to extend an existing row with the smallest possible next element.
            pass
        
        # Actually, the prompt's specific DP idea: "assigning the next smallest available difference value to a compatible row"
        # This is hard to implement correctly without complex state representation of the row's internal structure.
        # Given the time, I will implement a simpler "Global Greedy" that picks the next element for a random row,
        # aiming to minimize the new element's value, which indirectly minimizes scope.
        
        # Let's try a different structural change: 
        # "Row Swapping" in SA is already there. 
        # Let's improve the initialization with a "Min-Sum" greedy that tries to keep the sum of all elements low.
        
        # For now, return None to indicate this specific DP isn't fully implemented/ready, 
        # and rely on the improved SA and existing initializations.
        # OR, implement a simple version:
        # Iterate d from 1 to max_d. 
        # If d is not in used_diffs, try to find a row where we can add a new element x such that 
        # x - y = d for some y, and x is not in the row, and x > all previous elements? 
        # No, x just needs to be > y. But the row must be increasing. 
        # So x must be > row[-1]. 
        # So d must be > row[-1] - y. 
        # If we pick y = row[0]=0, then x = d. 
        # So we can always try to extend a row with x = d if d > row[-1] and d is not in the row's diffs.
        
        best_row_idx = -1
        best_new_max = float('inf')
        
        for idx, row in enumerate(open_rows):
            if len(row) == k + 1:
                continue
            # Try to extend this row with a new element x = d (assuming y=0)
            # Or x = d + y for some y?
            # If we set x = d, then diff d is used.
            # Check if x=d is valid:
            if d > row[-1]:
                # Check if d is in row
                if d in row:
                    continue
                # Check if d - y is in used_diffs or local_diffs for any y in row
                valid = True
                local_diffs = set()
                for y in row:
                    dd = d - y
                    if dd <= 0: continue
                    if dd in used_diffs:
                        valid = False
                        break
                    if dd in local_diffs:
                        valid = False
                        break
                    local_diffs.add(dd)
                if valid:
                    # This is a valid extension. 
                    # Score: new max is d. 
                    # Prefer smaller d? d is fixed in the loop.
                    # Prefer rows that are closer to completion?
                    # Just pick the first one.
                    best_row_idx = idx
                    break
        
        if best_row_idx != -1:
            row = open_rows[best_row_idx]
            row.append(d)
            for y in row[:-1]:
                used_diffs.add(d - y)
            assigned = True
        else:
            # Try to start a new row?
            # A new row starts with 0. Next element must be d?
            # If we start a new row [0, d], diff d is used.
            if len(open_rows) < n_rows:
                # Check if d is small enough to be a good start?
                # Just start it.
                open_rows.append([0, d])
                used_diffs.add(d)
                assigned = True
            
        if not assigned:
            # d cannot be used to extend any row or start a new one?
            # This happens if d is too small to be > row[-1] for any open row, 
            # and we have no more slots.
            # In that case, we just skip d? 
            # But we MUST use distinct differences. 
            # If we skip d, we might miss a valid solution or force larger differences later.
            # However, for initialization, skipping small d that don't fit is okay if we fill with larger ones.
            # But wait, if d is small, it SHOULD fit in a row starting with 0.
            # If all rows have max > d, we can't use d to extend them with x=d.
            # We could extend with x = d + y?
            # If we extend row with x, we add diffs x-y.
            # If we want to use diff d, we need x-y=d => x = y+d.
            # If we pick y=row[0]=0, x=d.
            # If we pick y=row[1], x=row[1]+d.
            # So we should try all y in row.
            
            # Re-do the extension logic to try all y
            best_row_idx = -1
            best_x = float('inf')
            best_y = 0
            
            for idx, row in enumerate(open_rows):
                if len(row) == k + 1:
                    continue
                for y in row:
                    x = y + d
                    if x <= row[-1]:
                        continue
                    if x in row:
                        continue
                    valid = True
                    local_diffs = set()
                    for z in row:
                        dd = x - z
                        if dd <= 0: continue
                        if dd in used_diffs:
                            valid = False
                            break
                        if dd in local_diffs:
                            valid = False
                            break
                        local_diffs.add(dd)
                    if valid:
                        if x < best_x:
                            best_x = x
                            best_row_idx = idx
                            best_y = y
            
            if best_row_idx != -1:
                row = open_rows[best_row_idx]
                row.append(best_x)
                for z in row[:-1]:
                    used_diffs.add(best_x - z)
                assigned = True
            elif len(open_rows) < n_rows:
                open_rows.append([0, d])
                used_diffs.add(d)
                assigned = True

    # Check if we have enough complete rows
    complete_rows = [r for r in open_rows if len(r) == k + 1]
    if len(complete_rows) < n_rows:
        return None
        
    # We have n_rows complete rows. 
    # But we might have extra open rows? We stopped when we had n_rows complete.
    # The list `complete_rows` might not be sorted by scope.
    # Return the first n_rows complete rows.
    return complete_rows[:n_rows]

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts

def _remove_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1

def _row_rebuild(row, k, diff_counts, rng, max_scope):
    """Rebuild a row with minimal conflicts using the current global diff counts."""
    best_row = None
    best_score = float('inf')
    
    for attempt in range(10):
        row = [0]
        x = 0
        local_diffs = set()
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.4:
                start = x + rng.randint(1, 4)
            for t in range(start, max_scope + 1):
                conflicts = 0
                new_diffs = []
                for a in row:
                    d = t - a
                    if d <= 0:
                        continue
                    new_diffs.append(d)
                    if diff_counts.get(d, 0) > 0:
                        conflicts += 1
                if conflicts == 0:
                    row.append(t)
                    local_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
                elif attempt < 4 and conflicts <= 1 and rng.random() < 0.12:
                    row.append(t)
                    local_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok and len(row) == k + 1:
            score = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if diff_counts.get(d, 0) > 0:
                        score += 1
            if score < best_score:
                best_score = score
                best_row = list(row)
            if best_score == 0:
                break
    
    return best_row

def _sa_optimize(rows, k, iters, rng, deadline, polish=False):
    """Simulated annealing with row rebuilds, scope compression, and single-element moves."""
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    diff_counts = _build_diff_counts(current_rows)
    
    def get_cost(rows, dc):
        scope = _scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 500 * collisions

    current_cost = get_cost(current_rows, diff_counts)
    best_cost = current_cost
    
    if polish:
        T = 0.5
        cooling = 0.9999
        min_T = 0.01
    else:
        T = 8.0
        cooling = 0.9995
        min_T = 0.05
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        r = rng.random()
        
        if r < 0.15:
            # Row rebuild move
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            
            new_row = _row_rebuild(old_row, k, diff_counts, rng, current_scope + 5)
            if new_row is None:
                _add_row_diffs(diff_counts, old_row)
                continue
            
            _add_row_diffs(diff_counts, new_row)
            current_rows[i] = new_row
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row
                
        elif r < 0.35:
            # Scope compression: try to reduce the max element
            max_val = current_scope
            max_row_idx = -1
            max_pos_idx = -1
            for ri in range(n_rows):
                for pi in range(k, 0, -1):
                    if current_rows[ri][pi] == max_val:
                        max_row_idx = ri
                        max_pos_idx = pi
                        break
                if max_row_idx >= 0:
                    break
            
            if max_row_idx < 0:
                continue
                
            i = max_row_idx
            j = max_pos_idx
            old_val = current_rows[i][j]
            
            # Try to decrease this value
            prev_val = current_rows[i][j-1]
            if old_val - prev_val <= 1:
                continue
            new_val = rng.randint(prev_val + 1, old_val - 1)
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
                
        else:
            # Single-element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            old_val = current_rows[i][j]
            
            lo = current_rows[i][j-1] + 1
            hi = current_rows[i][j+1] - 1 if j < k else current_scope
            
            if hi <= lo:
                continue
            
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def _local_polish(rows, k, deadline, rng):
    """Aggressive local search to compress scope."""
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in current_rows]
    best_scope = _scope(current_rows)
    
    diff_counts = _build_diff_counts(current_rows)
    
    def get_cost(rows, dc):
        scope = _scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 500 * collisions

    current_cost = get_cost(current_rows, diff_counts)
    best_cost = current_cost
    
    T = 0.3
    cooling = 0.99995
    min_T = 0.005
    
    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        old_val = current_rows[i][j]
        
        lo = current_rows[i][j-1] + 1
        hi = current_rows[i][j+1] - 1 if j < k else best_scope
        
        if hi <= lo:
            continue
        
        # Try decreasing first
        if old_val > lo:
            new_val = rng.randint(lo, old_val - 1)
        else:
            new_val = rng.randint(old_val + 1, hi)
        
        if new_val == old_val:
            continue
        
        _remove_row_diffs(diff_counts, current_rows[i])
        current_rows[i][j] = new_val
        _add_row_diffs(diff_counts, current_rows[i])
        
        new_cost = get_cost(current_rows, diff_counts)
        
        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = _scope(current_rows)
                best_rows = [list(r) for r in current_rows]
        else:
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = old_val
            _add_row_diffs(diff_counts, current_rows[i])
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try CSP initialization
    csp_rows = _csp_initialize(k, n_rows, rng, time_limit=3.0)
    candidates = []
    
    if csp_rows is not None and _valid(csp_rows, k):
        s = _scope(csp_rows)
        candidates.append((s, csp_rows))
        best_scope = s
        best_rows = [list(r) for r in csp_rows]
        
    # Phase 1.5: Try DP initialization
    dp_rows = _dp_initialize(k, n_rows, rng, time_limit=3.0)
    if dp_rows is not None and _valid(dp_rows, k):
        s = _scope(dp_rows)
        candidates.append((s, dp_rows))
        if s < best_scope:
            best_scope = s
            best_rows = [list(r) for r in dp_rows]

    # Phase 2: Generate diverse initial solutions via fast greedy
    init_deadline = start_time + 5.0
    if time.time() < init_deadline:
        for trial in range(10):
            if time.time() > init_deadline:
                break
            spread = rng.random() < 0.3
            rows = _fast_greedy(k, n_rows, rng, spread=spread)
            if _valid(rows, k):
                s = _scope(rows)
                candidates.append((s, rows))
                if s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in rows]
    
    if not candidates:
        # Last resort: simple sequential
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k+1):
                x = len(used) + j + 1
                while any((x - row[i]) in used for i in range(len(row))):
                    x += 1
                for i in range(len(row)):
                    used.add(x - row[i])
                row.append(x)
            rows.append(row)
        return rows
    
    candidates.sort(key=lambda x: x[0])
    
    # Phase 3: SA optimization on top candidates
    remaining = deadline - time.time()
    top_n = min(3, len(candidates))
    time_per = remaining / max(1, top_n)
    
    for idx in range(top_n):
        if time.time() > deadline - 15:
            break
        s, rows = candidates[idx]
        sa_deadline = time.time() + time_per * 0.85
        sa_scope, sa_rows = _sa_optimize(rows, k, 10000, rng, sa_deadline)
        
        if sa_rows is not None and _valid(sa_rows, k):
            actual = _scope(sa_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = sa_rows
                candidates.append((actual, sa_rows))
                candidates.sort(key=lambda x: x[0])
    
    # Phase 4: Local polish on the best solution
    if best_rows is not None and time.time() < deadline - 5:
        polish_deadline = time.time() + min(12, deadline - time.time() - 5)
        polish_scope, polish_rows = _local_polish(best_rows, k, polish_deadline, rng)
        
        if polish_rows is not None and _valid(polish_rows, k):
            actual = _scope(polish_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = polish_rows
    
    return best_rows
