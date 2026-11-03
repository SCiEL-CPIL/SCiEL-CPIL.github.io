import time
import random
import math

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

def _greedy_construct(n_rows, k, rng, bias=1.0):
    """Per-row greedy construction with randomized bias."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            candidates = []
            x = last + 1
            max_x = last + 100
            while x <= max_x and len(candidates) < 20:
                valid = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    candidates.append((x, new_diffs))
                x += 1
            if not candidates:
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        candidates.append((x, new_diffs))
                        break
                    x += 1
            weights = [1.0 / (c[0] - last + 1) ** bias for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = c
                    break
            x_val, new_diffs = chosen
            row.append(x_val)
            last = x_val
            used.update(new_diffs)
        rows.append(row)
    return rows

def _backtrack_row(row_idx, pos, current_row, last_val, used, k, max_scope, rng, candidates_limit=8):
    """
    Backtracking to fill positions pos..k of a single row.
    Returns list of values for positions pos..k, or None if stuck.
    """
    if pos == k + 1:
        return []
    
    start = last_val + 1
    candidates = []
    
    for v in range(start, max_scope + 1):
        valid = True
        new_diffs = []
        for i in range(len(current_row)):
            d = v - current_row[i]
            if d in used:
                valid = False
                break
            new_diffs.append(d)
        if valid:
            candidates.append((v, new_diffs))
            if len(candidates) >= candidates_limit:
                break
    
    if not candidates:
        return None
    
    # Try candidates in random order with bias toward smaller values
    # Weight by 1/v^0.5 to prefer smaller but allow exploration
    weights = [1.0 / math.sqrt(c[0]) for c in candidates]
    total_w = sum(weights)
    
    # Shuffle with weights
    order = list(range(len(candidates)))
    for i in range(len(order) - 1, 0, -1):
        r = rng.random() * sum(weights[j] for j in order[i:])
        cum = 0
        for j in range(i, -1, -1):
            cum += weights[order[j]]
            if r <= cum:
                order[i], order[j] = order[j], order[i]
                break
    
    for idx in order:
        v, new_diffs = candidates[idx]
        current_row.append(v)
        used.update(new_diffs)
        
        result = _backtrack_row(row_idx, pos + 1, current_row, v, used, k, max_scope, rng, candidates_limit)
        
        if result is not None:
            return [v] + result
        else:
            current_row.pop()
            used.difference_update(new_diffs)
    
    return None

def _construct_with_backtrack(n_rows, k, rng, max_scope=150):
    """
    Construct all rows using backtracking for each row.
    """
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        current_row = [0]
        last_val = 0
        
        result = _backtrack_row(row_idx, 1, current_row, last_val, used, k, max_scope, rng)
        
        if result is None:
            # Fallback to greedy for this row
            row = [0]
            last = 0
            for j in range(1, k + 1):
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        break
                    x += 1
                row.append(x)
                last = x
                used.update(new_diffs)
            rows.append(row)
        else:
            row = current_row + result
            rows.append(row)
    
    return rows

def _sa_optimize(rows, k, deadline, rng, T0=10.0, cooling=0.9995, min_T=0.1):
    """
    Simulated annealing optimization with efficient diff tracking.
    """
    state_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Build used diffs: diff -> (row_idx, i, j)
    used_diffs = {}
    for ri, row in enumerate(state_rows):
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                used_diffs[d] = (ri, i, j)
    
    T = T0
    n_rows = len(state_rows)
    
    while time.time() < deadline:
        # Pick random row and position
        ri = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)
        row = state_rows[ri]
        old_val = row[pos]
        
        # Propose new value
        if rng.random() < 0.75:
            # Try to decrease
            max_drop = max(1, old_val - row[pos-1] - 1)
            if max_drop > 0:
                drop = rng.randint(1, min(max_drop, max(1, old_val // 4)))
                new_val = old_val - drop
            else:
                T = max(T * cooling, min_T)
                continue
        else:
            # Try to increase slightly
            delta = rng.randint(1, 3)
            new_val = old_val + delta
        
        # Bound check
        lo = row[pos-1] + 1
        hi = row[pos+1] - 1 if pos < k else 200
        new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            T = max(T * cooling, min_T)
            continue
        
        # Check for conflicts
        conflicts = 0
        for t in range(k + 1):
            if t == pos:
                continue
            d = abs(new_val - row[t])
            if d in used_diffs:
                # Check if it's a diff we're removing
                old_d = abs(old_val - row[t])
                if d != old_d:
                    conflicts += 1
        
        if conflicts > 0:
            T = max(T * cooling, min_T)
            continue
        
        # Apply move
        row[pos] = new_val
        old_scope = _scope(state_rows)
        
        # Update used_diffs
        for t in range(k + 1):
            if t == pos:
                continue
            old_d = abs(old_val - row[t])
            if old_d in used_diffs:
                del used_diffs[old_d]
        for t in range(k + 1):
            if t == pos:
                continue
            new_d = abs(new_val - row[t])
            used_diffs[new_d] = (ri, min(pos, t), max(pos, t))
        
        new_scope = _scope(state_rows)
        delta = new_scope - old_scope
        
        if delta <= 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
            # Accept
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in state_rows]
        else:
            # Revert
            row[pos] = old_val
            for t in range(k + 1):
                if t == pos:
                    continue
                new_d = abs(new_val - row[t])
                if new_d in used_diffs:
                    del used_diffs[new_d]
            for t in range(k + 1):
                if t == pos:
                    continue
                old_d = abs(old_val - row[t])
                used_diffs[old_d] = (ri, min(pos, t), max(pos, t))
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate diverse initial solutions
    init_deadline = start_time + 15.0
    candidates = []
    
    # Try backtracking-based construction first
    for trial in range(20):
        if time.time() > init_deadline:
            break
        max_scope = 120 + trial * 5
        rows = _construct_with_backtrack(n_rows, k, rng, max_scope=max_scope)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    # Also try greedy construction for diversity
    for trial in range(15):
        if time.time() > init_deadline:
            break
        bias = rng.uniform(0.3, 2.5)
        rows = _greedy_construct(n_rows, k, rng, bias=bias)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    if not candidates or best_rows is None:
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
    
    # Phase 2: SA optimization on top candidates
    remaining = deadline - time.time()
    top_n = min(6, len(candidates))
    time_per = remaining / max(1, top_n)
    
    for idx in range(top_n):
        if time.time() > deadline - 10:
            break
        s, rows = candidates[idx]
        sa_deadline = time.time() + time_per * 0.85
        sa_scope, sa_rows = _sa_optimize(rows, k, sa_deadline, rng)
        
        if sa_rows is not None and _valid(sa_rows, k):
            actual = _scope(sa_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = sa_rows
                candidates.append((actual, sa_rows))
                candidates.sort(key=lambda x: x[0])
    
    # Phase 3: Final polish with local search
    if best_rows is not None and time.time() < deadline - 5:
        polish_deadline = time.time() + min(8, deadline - time.time() - 5)
        for _ in range(200):
            if time.time() > polish_deadline:
                break
            improved = False
            for ri in range(n_rows):
                row = best_rows[ri]
                for j in range(k, 0, -1):
                    lo = row[j-1] + 1
                    for new_val in range(lo, row[j]):
                        old = row[j]
                        row[j] = new_val
                        if _valid(best_rows, k):
                            ns = _scope(best_rows)
                            if ns < best_scope:
                                best_scope = ns
                                improved = True
                            else:
                                row[j] = old
                            break
                        else:
                            row[j] = old
            if not improved:
                break
    
    return best_rows
