import time
import random
import math
import sys
from collections import Counter

_BEST = None
_DEADLINE = 0.0

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
            d = row[j] - r[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - r[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1

def _row_rebuild(row, k, diff_counts, rng, max_scope):
    best_row = None
    best_score = float('inf')
    
    for attempt in range(12):
        row = [0]
        x = 0
        local_diffs = set()
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.4:
                start = x + rng.randint(1, 4)
            
            best_t = None
            best_conflicts = float('inf')
            for t in range(start, max_scope + 1):
                conflicts = 0
                for a in row:
                    d = t - a
                    if d <= 0:
                        continue
                    if diff_counts.get(d, 0) > 0:
                        conflicts += 1
                        if conflicts >= best_conflicts:
                            break
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_t = t
                    if conflicts == 0:
                        break
            
            if best_t is not None:
                if best_conflicts == 0 or (attempt < 4 and best_conflicts <= 1 and rng.random() < 0.2):
                    row.append(best_t)
                    for a in row[:-1]:
                        d = best_t - a
                        if d > 0:
                            local_diffs.add(d)
                    x = best_t
                    found = True
        
            if not found:
                ok = False
                break
        if ok and len(row) == k + 1:
            score = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - r[i]
                    if diff_counts.get(d, 0) > 0:
                        score += 1
            if score < best_score:
                best_score = score
                best_row = list(row)
            if best_score == 0:
                break
    
    return best_row

def _dfs_row_minimize(row, k, diff_counts, max_scope, rng, deadline):
    best_row = None
    best_max = max_scope + 1
    
    def dfs(pos, current_row, current_x):
        nonlocal best_row, best_max
        if time.time() > deadline:
            return
        if pos == k + 1:
            if current_x < best_max:
                best_max = current_x
                best_row = list(current_row)
            return
        if current_x + (k + 1 - pos) >= best_max:
            return
        start = current_x + 1
        for t in range(start, best_max):
            ok = True
            for a in current_row:
                d = t - a
                if d <= 0:
                    continue
                if diff_counts.get(d, 0) > 0:
                    ok = False
                    break
            if ok:
                current_row.append(t)
                dfs(pos + 1, current_row, t)
                current_row.pop()
            if time.time() > deadline:
                return
    
    dfs(1, [0], 0)
    return best_row

def _sa_optimize(rows, k, iters, rng, deadline, polish=False):
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    diff_counts = _build_diff_counts(current_rows)
    
    def get_cost(rows, dc):
        scope = _scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 20 * collisions
    
    current_cost = get_cost(current_rows, diff_counts)
    best_cost = current_cost
    
    if polish:
        T = 0.3
        cooling = 0.99995
        min_T = 0.005
    else:
        T = 15.0
        cooling = 0.9995
        min_T = 0.1
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        r = rng.random()
        
        if r < 0.20:
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            
            new_row = _row_rebuild(old_row, k, diff_counts, rng, current_scope + 3)
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
                
        elif r < 0.40:
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
            
            prev_val = current_rows[i][j-1]
            new_val = rng.randint(prev_val + 1, old_val - 1)
            
            if new_val <= prev_val or new_val >= old_val:
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
                
        else:
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else current_scope + 10
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                continue
            
            if old_val == current_scope:
                new_val = rng.randint(lo, min(hi, old_val - 1))
            else:
                if rng.random() < 0.7:
                    span = hi - lo
                    delta = int(rng.gauss(0, max(1, span * 0.2)))
                    new_val = old_val + delta
                else:
                    new_val = rng.randint(lo, hi)
            
            if new_val == old_val or new_val < lo or new_val > hi:
                continue
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_scope = _scope(current_rows)
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
            
        T = max(T * cooling, min_T)
        
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 108.0
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    for _ in range(4):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 8)
                for t in range(x + 1, x + 1 + max_step * 3):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)
        
    candidates.sort(key=_scope)
    
    for idx, cand in enumerate(candidates[:2]):
        if time.time() > _DEADLINE:
            break
        remaining = _DEADLINE - time.time()
        if idx == 0:
            sa_deadline = time.time() + remaining * 0.4
        else:
            sa_deadline = _DEADLINE
        iters = int((sa_deadline - time.time()) * 8000)
        iters = max(iters, 2000)
        s, rows = _sa_optimize(cand, k, iters, rng, sa_deadline)
        consider(rows)
        
    if _BEST is not None and time.time() < _DEADLINE:
        best_rows = [list(r) for r in _BEST[1]]
        diff_counts = _build_diff_counts(best_rows)
        current_scope = _scope(best_rows)
        improved = True
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            for i in range(n_rows):
                if time.time() > _DEADLINE - 5:
                    break
                old_row = list(best_rows[i])
                _remove_row_diffs(diff_counts, old_row)
                new_row = _dfs_row_minimize(old_row, k, diff_counts, current_scope, rng, _DEADLINE - 5)
                if new_row is not None and new_row[-1] < old_row[-1]:
                    best_rows[i] = new_row
                    _add_row_diffs(diff_counts, new_row)
                    current_scope = min(current_scope, _scope(best_rows))
                    improved = True
                else:
                    _add_row_diffs(diff_counts, old_row)
                    best_rows[i] = old_row
        consider(best_rows)
    
    if _BEST is not None and time.time() < _DEADLINE:
        remaining_time = _DEADLINE - time.time()
        extra_iters = int(remaining_time * 10000)
        if extra_iters > 500:
            s, rows = _sa_optimize(_BEST[1], k, extra_iters, rng, _DEADLINE, polish=True)
            consider(rows)
            
    if _BEST is not None and time.time() < _DEADLINE - 3:
        best_rows = [list(r) for r in _BEST[1]]
        diff_counts = _build_diff_counts(best_rows)
        current_scope = _scope(best_rows)
        for i in range(n_rows):
            if time.time() > _DEADLINE - 3:
                break
            old_row = list(best_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            new_row = _dfs_row_minimize(old_row, k, diff_counts, current_scope, rng, _DEADLINE - 3)
            if new_row is not None and new_row[-1] < old_row[-1]:
                best_rows[i] = new_row
                _add_row_diffs(diff_counts, new_row)
                current_scope = min(current_scope, _scope(best_rows))
            else:
                _add_row_diffs(diff_counts, old_row)
                best_rows[i] = old_row
        consider(best_rows)
    
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
