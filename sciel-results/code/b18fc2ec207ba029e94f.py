"""Difference triangle set (4,7) - structured local search with fast updates."""
import time
import random
import math
import sys

_DEADLINE = 150.0
_SAFETY = 10.0
_TIME_LIMIT = _DEADLINE - _SAFETY


def _row_diffs(row):
    """All positive differences within a single row."""
    d = []
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d.append(row[j] - ri)
    return d


def _is_valid(rows):
    used = set()
    for r in rows:
        for d in _row_diffs(r):
            if d in used:
                return False
            used.add(d)
    return True


def _score(rows):
    """Objective: largest entry (lower better)."""
    return max(max(r) for r in rows)


def _greedy_init(n_rows, k):
    """Greedy construction with randomized ordering."""
    rng = random.Random(42)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _greedy_init_random(n_rows, k, seed):
    """Randomized greedy: try different starting points."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        attempts = 0
        while len(row) < k + 1 and attempts < 10000:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                attempts = 0
            else:
                attempts += 1
        if len(row) < k + 1:
            # fallback: keep searching
            x = max(row)
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows


def _build_used(rows):
    """Build the global used-difference set."""
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(r[j] - r[i])
    return used


def _local_search(rows, k, n_rows, deadline):
    """Fast local search with incremental updates."""
    used = _build_used(rows)
    best = _score(rows)
    best_rows = [list(r) for r in rows]
    best_used = set(used)
    
    # Track per-row max for fast score updates
    row_maxs = [max(r) for r in rows]
    global_max = max(row_maxs)
    
    T = 50.0
    T_min = 0.1
    cool = 0.99995
    
    rng = random.Random(999)
    
    # Precompute row lengths
    k1 = k + 1
    
    while time.time() < deadline:
        # Pick random row and position
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k1)
        row = rows[ri]
        
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi + 1 < k1 else global_max + 30
        if hi < lo:
            hi = lo
        
        nv = rng.randrange(lo, hi + 1)
        if nv == row[pi]:
            continue
        
        old = row[pi]
        
        # Compute affected differences
        affected_old = set()
        affected_new = set()
        for i in range(k1):
            if i != pi:
                affected_old.add(abs(old - row[i]))
                affected_new.add(abs(nv - row[i]))
        
        removed = affected_old - affected_new
        added = affected_new - affected_old
        
        # Check for collisions
        ok = True
        for d in added:
            if d in used and d not in removed:
                ok = False
                break
        if not ok:
            continue
        
        # Compute score change
        old_row_max = row_maxs[ri]
        new_row_max = max(nv, row[-1]) if nv > row[-1] else row[-1]
        # Actually new_row_max = max of the new row
        if nv >= old_row_max:
            new_row_max = nv
        else:
            new_row_max = old_row_max
        
        if new_row_max != old_row_max:
            if old_row_max == global_max and new_row_max < global_max:
                # This row was the max and now isn't
                others = [row_maxs[j] for j in range(n_rows) if j != ri]
                new_global_max = max(others) if others else 0
                delta = new_global_max - global_max
            elif new_row_max > global_max:
                delta = new_row_max - global_max
            else:
                delta = 0
        else:
            delta = 0
        
        # Accept/reject
        accept = False
        if delta <= 0:
            accept = True
        else:
            if T > T_min:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            row[pi] = nv
            for d in removed:
                used.discard(d)
            for d in added:
                used.add(d)
            row_maxs[ri] = new_row_max
            if new_row_max > global_max:
                global_max = new_row_max
            elif old_row_max == global_max and new_row_max < global_max:
                # Recompute global max
                global_max = max(row_maxs)
            
            if global_max < best:
                best = global_max
                best_rows = [list(r) for r in rows]
                best_used = set(used)
        
        T *= cool
        if T < T_min:
            T = T_min
        
        # Occasionally restart temperature
        if rng.random() < 0.001:
            T = min(T * 10, 50.0)
    
    return best_rows, best


def _multi_start_local_search(n_rows, k, deadline):
    """Try multiple greedy initializations and local search each."""
    best_rows = None
    best_score = float('inf')
    
    seeds = [42, 123, 456, 789, 321, 654, 987, 135, 246, 738]
    idx = 0
    
    while time.time() < deadline and idx < len(seeds):
        rows = _greedy_init_random(n_rows, k, seeds[idx])
        idx += 1
        
        if time.time() < deadline - 5:
            ls_rows, ls_score = _local_search([list(r) for r in rows], k, n_rows, deadline)
            if ls_score < best_score:
                best_score = ls_score
                best_rows = ls_rows
        else:
            if _score(rows) < best_score:
                best_score = _score(rows)
                best_rows = [list(r) for r in rows]
    
    return best_rows, best_score


def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT
    
    # Quick greedy init
    rows = _greedy_init(n_rows, k)
    best_rows = [list(r) for r in rows]
    best = _score(best_rows)
    
    # Multi-start local search
    if time.time() < deadline - 10:
        ls_rows, ls_best = _multi_start_local_search(n_rows, k, deadline)
        if ls_best < best:
            best = ls_best
            best_rows = ls_rows
    
    # Final validity check
    if not _is_valid(best_rows):
        best_rows = _greedy_init(n_rows, k)
    
    return best_rows
