import time
import random
import math
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
            d = row[j] - row[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1

def _row_rebuild(row, k, diff_counts, rng, max_scope, target_scope=None):
    best_row = None
    best_score = float('inf')
    
    effective_max = max_scope if target_scope is None else min(max_scope, target_scope)
    
    for attempt in range(12):
        row = [0]
        x = 0
        local_diffs = set()
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.5:
                start = x + rng.randint(1, 5)
            for t in range(start, effective_max + 1):
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
                elif attempt < 4 and conflicts <= 1 and rng.random() < 0.2:
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
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    diff_counts = _build_diff_counts(current_rows)
    
    def get_cost(rows, dc, scope):
        collisions = sum(max(0, c - 1) for c in dc.values())
        # Smooth penalty: use log-sum-exp style for collision count
        if collisions > 0:
            collision_penalty = 100 * collisions + 10 * collisions * collisions
        else:
            collision_penalty = 0
        return scope + collision_penalty
    
    current_cost = get_cost(current_rows, diff_counts, current_scope)
    best_cost = current_cost
    
    if polish:
        T = 0.3
        cooling = 0.99995
        min_T = 0.005
    else:
        T = 5.0
        cooling = 0.9997
        min_T = 0.1
    
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        r = rng.random()
        
        if r < 0.12:
            # Row rebuild move
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            
            new_row = _row_rebuild(old_row, k, diff_counts, rng, current_scope + 3, current_scope)
            if new_row is None:
                _add_row_diffs(diff_counts, old_row)
                continue
            
            _add_row_diffs(diff_counts, new_row)
            current_rows[i] = new_row
            new_scope = _scope(current_rows)
            new_cost = get_cost(current_rows, diff_counts, new_scope)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row
                
        elif r < 0.25:
            # Targeted scope compression: find the row with the max element and try to reduce it
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
            
            # Try multiple candidate values
            prev_val = current_rows[i][j-1]
            candidates = []
            for _ in range(5):
                new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val - 1))
                if new_val != old_val:
                    candidates.append(new_val)
            
            if not candidates:
                continue
            
            best_candidate = None
            best_candidate_cost = float('inf')
            
            for new_val in candidates:
                if new_val <= prev_val or new_val >= old_val:
                    continue
                
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = new_val
                _add_row_diffs(diff_counts, current_rows[i])
                
                new_scope = _scope(current_rows)
                cand_cost = get_cost(current_rows, diff_counts, new_scope)
                
                if cand_cost < best_candidate_cost:
                    best_candidate_cost = cand_cost
                    best_candidate = new_val
                    best_candidate_scope = new_scope
                
                current_rows[i][j] = old_val
                _remove_row_diffs(diff_counts, current_rows[i])
                _add_row_diffs(diff_counts, current_rows[i])
            
            if best_candidate is not None:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = best_candidate
                _add_row_diffs(diff_counts, current_rows[i])
                
                new_scope = _scope(current_rows)
                new_cost = get_cost(current_rows, diff_counts, new_scope)
                
                if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                    current_cost = new_cost
                    current_scope = new_scope
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = current_scope
                        best_rows = [list(r) for r in current_rows]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    _remove_row_diffs(diff_counts, current_rows[i])
                    current_rows[i][j] = old_val
                    _add_row_diffs(diff_counts, current_rows[i])
                    
        elif r < 0.45:
            # Pair move: adjust two elements in the same row
            i = rng.randint(0, n_rows - 1)
            j1 = rng.randint(1, k)
            j2 = rng.randint(1, k)
            if j1 == j2:
                j2 = (j1 % k) + 1
            
            old_v1 = current_rows[i][j1]
            old_v2 = current_rows[i][j2]
            
            lo1 = current_rows[i][j1-1] + 1
            hi1 = (current_rows[i][j1+1] - 1 if j1 < k else current_scope + 10)
            lo2 = current_rows[i][j2-1] + 1
            hi2 = (current_rows[i][j2+1] - 1 if j2 < k else current_scope + 10)
            
            if lo1 > hi1 or lo2 > hi2:
                continue
            
            new_v1 = rng.randint(lo1, hi1)
            new_v2 = rng.randint(lo2, hi2)
            
            if new_v1 == old_v1 and new_v2 == old_v2:
                continue
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j1] = new_v1
            current_rows[i][j2] = new_v2
            
            # Check if row is still strictly increasing
            valid_row = True
            for jj in range(1, k + 1):
                if current_rows[i][jj] <= current_rows[i][jj-1]:
                    valid_row = False
                    break
            
            if not valid_row:
                current_rows[i][j1] = old_v1
                current_rows[i][j2] = old_v2
                _remove_row_diffs(diff_counts, current_rows[i])
                _add_row_diffs(diff_counts, current_rows[i])
                continue
            
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_scope = _scope(current_rows)
            new_cost = get_cost(current_rows, diff_counts, new_scope)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j1] = old_v1
                current_rows[i][j2] = old_v2
                _add_row_diffs(diff_counts, current_rows[i])
                
        else:
            # Single-element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else current_scope + 10
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                continue
            
            if rng.random() < 0.6:
                span = hi - lo
                delta = int(rng.gauss(0, max(1, span * 0.15)))
                new_val = old_val + delta
            else:
                new_val = rng.randint(lo, hi)
            
            if new_val == old_val or new_val < lo or new_val > hi:
                continue
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_scope = _scope(current_rows)
            new_cost = get_cost(current_rows, diff_counts, new_scope)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
            
        T = max(T * cooling, min_T)
        
        # Restart if stuck
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            T = max(T * 2, 0.5)
            
    return best_scope, best_rows

def _local_polish(rows, k, deadline, rng):
    """Final polish: try to reduce scope by 1, 2, 3... using targeted row rebuilds."""
    rows = [list(r) for r in rows]
    scope = _scope(rows)
    diff_counts = _build_diff_counts(rows)
    n_rows = len(rows)
    
    for target in range(scope - 1, max(scope - 10, 0), -1):
        if time.time() > deadline:
            break
        
        improved = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Find rows that have elements >= target + 1
            candidates = []
            for i in range(n_rows):
                if any(v > target for v in rows[i]):
                    candidates.append(i)
            
            if not candidates:
                break
            
            # Try to rebuild each candidate row
            for i in candidates:
                if time.time() > deadline:
                    break
                
                old_row = list(rows[i])
                _remove_row_diffs(diff_counts, old_row)
                
                new_row = _row_rebuild(old_row, k, diff_counts, rng, target, target)
                
                if new_row is not None:
                    _add_row_diffs(diff_counts, new_row)
                    rows[i] = new_row
                    improved = True
                else:
                    _add_row_diffs(diff_counts, old_row)
            
            if not improved:
                break
        
        # Check if we achieved the target
        current_scope = _scope(rows)
        if current_scope <= target:
            scope = current_scope
            break
        else:
            # Revert to best known
            break
    
    return scope, rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 105.0
    
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
    
    for _ in range(6):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 10)
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
            rows.append(rows)
        candidates.append(rows)
        
    candidates.sort(key=_scope)
    
    for idx, cand in enumerate(candidates[:3]):
        if time.time() > _DEADLINE:
            break
        remaining = _DEADLINE - time.time()
        if idx == 0:
            sa_deadline = time.time() + remaining * 0.35
        elif idx == 1:
            sa_deadline = time.time() + remaining * 0.35
        else:
            sa_deadline = _DEADLINE
        iters = int((sa_deadline - time.time()) * 6000)
        iters = max(iters, 1500)
        s, rows = _sa_optimize(cand, k, iters, rng, sa_deadline)
        consider(rows)
        
    if _BEST is not None and time.time() < _DEADLINE:
        remaining_time = _DEADLINE - time.time()
        if remaining_time > 8:
            # Polish pass with low temperature
            polish_deadline = time.time() + remaining_time * 0.5
            polish_iters = int((polish_deadline - time.time()) * 8000)
            polish_iters = max(polish_iters, 300)
            s, rows = _sa_optimize(_BEST[1], k, polish_iters, rng, polish_deadline, polish=True)
            consider(rows)
            
            # Local polish: try to reduce scope by small amounts
            if time.time() < _DEADLINE - 3:
                local_deadline = _DEADLINE - 3
                s, rows = _local_polish(_BEST[1], k, local_deadline, rng)
                consider(rows)
            
            # Final fine-tuning
            if time.time() < _DEADLINE - 2:
                final_deadline = _DEADLINE - 2
                final_iters = int((final_deadline - time.time()) * 10000)
                final_iters = max(final_iters, 200)
                s, rows = _sa_optimize(_BEST[1], k, final_iters, rng, final_deadline, polish=True)
                consider(rows)
            
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
