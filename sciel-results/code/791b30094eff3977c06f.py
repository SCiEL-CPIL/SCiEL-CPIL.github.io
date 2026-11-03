import time
import random
import math
import sys

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


def _row_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _build_used_set(rows):
    used = set()
    for r in rows:
        used |= _row_diffs(r)
    return used


def _greedy_construct(n_rows, k, rng, max_scope, spread_factor=1.0, randomize=False):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            found = False
            step = 1
            if randomize:
                step = rng.randint(1, 3)
            for t in range(x + step, max_scope + 1):
                new_diffs = set(t - a for a in row)
                if not (new_diffs & used):
                    row.append(t)
                    used |= new_diffs
                    x = t
                    found = True
                    break
            if not found:
                # Try from x+1
                for t in range(x + 1, max_scope + 1):
                    new_diffs = set(t - a for a in row)
                    if not (new_diffs & used):
                        row.append(t)
                        used |= new_diffs
                        x = t
                        found = True
                        break
            if not found:
                return None
        rows.append(row)
    return rows


def _local_search(rows, k, deadline, rng):
    """
    Local search to minimize scope. Uses set-based difference tracking.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    best = [list(r) for r in current]
    best_scope = current_scope
    
    used = _build_used_set(current)
    
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    def try_move_row(i, new_row):
        nonlocal used, current, current_scope, best, best_scope
        old_row = current[i]
        old_diffs = _row_diffs(old_row)
        new_diffs = _row_diffs(new_row)
        
        # Check conflicts
        if new_diffs & used:
            return False
        
        # Remove old, add new
        for d in old_diffs:
            if d in new_diffs:
                continue
            used.discard(d)
        for d in new_diffs:
            if d in old_diffs:
                continue
            used.add(d)
        
        current[i] = new_row
        new_scope = _scope(current)
        return True, new_scope, old_row, old_diffs, new_diffs
    
    def revert_move(i, old_row, old_diffs, new_diffs):
        nonlocal used, current
        for d in new_diffs:
            if d in old_diffs:
                continue
            used.discard(d)
        for d in old_diffs:
            if d in new_diffs:
                continue
            used.add(d)
        current[i] = old_row
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        if iters % 10000 == 0:
            T = max(T * cooling, min_T)
        
        r = rng.random()
        
        if r < 0.3:
            # Single element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current[i][j]
            prev_val = current[i][j - 1]
            next_val = current[i][j + 1] if j < k else current_scope + 20
            
            lo = prev_val + 1
            hi = next_val - 1
            if lo > hi:
                continue
            
            # Try to move element
            candidates = []
            # Focus on reducing scope
            if current[i][j] > best_scope:
                candidates.extend(range(lo, min(old_val, best_scope + 1)))
            # Random moves
            for _ in range(5):
                candidates.append(rng.randint(lo, hi))
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                if new_val < lo or new_val > hi:
                    continue
                
                old_row = list(current[i])
                current[i][j] = new_val
                new_row = current[i]
                
                old_diffs = _row_diffs(old_row)
                new_diffs = _row_diffs(new_row)
                
                conflict = False
                for d in new_diffs:
                    if d in old_diffs:
                        continue
                    if d in used:
                        conflict = True
                        break
                if conflict:
                    current[i] = old_row
                    continue
                
                # Accept
                for d in old_diffs:
                    if d in new_diffs:
                        continue
                    used.discard(d)
                for d in new_diffs:
                    if d in old_diffs:
                        continue
                    used.add(d)
                
                new_scope = _scope(current)
                delta = new_scope - current_scope
                if delta < 0 or rng.random() < math.exp(-delta / max(T, min_T)):
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [list(r) for r in current]
                else:
                    # Revert
                    for d in new_diffs:
                        if d in old_diffs:
                            continue
                        used.discard(d)
                    for d in old_diffs:
                        if d in new_diffs:
                            continue
                        used.add(d)
                    current[i] = old_row
                break
        
        else:
            # Row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row = list(current[i])
            old_diffs = _row_diffs(old_row)
            
            # Remove from used
            for d in old_diffs:
                used.discard(d)
            
            # Try to rebuild row
            best_new_row = None
            best_new_scope = float('inf')
            
            for attempt in range(10):
                new_row = [0]
                x = 0
                ok = True
                for j in range(1, k + 1):
                    found = False
                    start = x + 1
                    if attempt > 0:
                        start = x + rng.randint(1, 3)
                    for t in range(start, best_scope + 10):
                        new_diffs = set(t - a for a in new_row)
                        if not (new_diffs & used):
                            new_row.append(t)
                            x = t
                            found = True
                            break
                    if not found:
                        ok = False
                        break
                if ok:
                    s = new_row[-1]
                    if s < best_new_scope:
                        best_new_scope = s
                        best_new_row = list(new_row)
                    if best_new_scope < best_scope:
                        break
            
            if best_new_row is not None:
                new_diffs = _row_diffs(best_new_row)
                current[i] = best_new_row
                for d in new_diffs:
                    used.add(d)
                
                new_scope = _scope(current)
                delta = new_scope - current_scope
                if delta < 0 or rng.random() < math.exp(-delta / max(T, min_T)):
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [list(r) for r in current]
                else:
                    # Revert
                    for d in new_diffs:
                        used.discard(d)
                    current[i] = old_row
                    for d in old_diffs:
                        used.add(d)
            else:
                # Revert
                current[i] = old_row
                for d in old_diffs:
                    used.add(d)
    
    return best, best_scope


def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 110.0
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
    
    rng = random.Random(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Baseline greedy
    for spread in [0.8, 1.0, 1.2, 1.5]:
        rows = _greedy_construct(n_rows, k, rng, 200, spread_factor=spread)
        if rows:
            candidates.append(rows)
    
    # Randomized greedy
    for _ in range(3):
        rows = _greedy_construct(n_rows, k, rng, 200, randomize=True)
        if rows:
            candidates.append(rows)
    
    # Sort by scope
    candidates.sort(key=_scope)
    
    # Try each candidate with local search
    for i, cand in enumerate(candidates):
        if time.time() > _DEADLINE - 10:
            break
        
        remaining = _DEADLINE - time.time()
        # Allocate time: more for better candidates
        if i == 0:
            t_alloc = min(30.0, remaining * 0.5)
        elif i == 1:
            t_alloc = min(20.0, remaining * 0.3)
        else:
            t_alloc = min(10.0, remaining * 0.2)
        
        if t_alloc < 2.0:
            break
        
        deadline = time.time() + t_alloc
        best_rows, best_scope = _local_search(cand, k, deadline, rng)
        consider(best_rows)
        
        # If we found a valid solution with scope < 112, we're doing well
        if _BEST is not None and _BEST[0] <= 112:
            break
    
    if _BEST is not None:
        return _BEST[1]
    
    # Fallback: return the best candidate
    return candidates[0]
