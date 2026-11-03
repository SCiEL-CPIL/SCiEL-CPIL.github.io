import time
import random
import math
from collections import Counter, defaultdict

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
    """Return list of all positive differences in a row."""
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

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

def _build_diff_map(rows):
    """Map each difference to the list of (row_idx, i, j) that produce it."""
    dmap = defaultdict(list)
    for ri, r in enumerate(rows):
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                dmap[d].append((ri, i, j))
    return dmap

def _conflicts(dmap):
    """Return list of differences that appear more than once."""
    return [d for d, lst in dmap.items() if len(lst) > 1]

def _n_conflicts(dmap):
    return sum(max(0, len(lst) - 1) for lst in dmap.values())

def _cost(rows, dmap):
    scope = _scope(rows)
    nc = _n_conflicts(dmap)
    return scope + 1000 * nc

def _sa_optimize_conflict_driven(rows, k, iters, rng, deadline):
    """SA with conflict-driven repair: when conflicts exist, target conflicting elements."""
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    dmap = _build_diff_map(current_rows)
    current_cost = _cost(current_rows, dmap)
    best_cost = current_cost
    
    T = 5.0
    cooling = 0.9997
    min_T = 0.1
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        # Decide move type based on conflict state
        nc = _n_conflicts(dmap)
        
        if nc > 0 and rng.random() < 0.6:
            # Conflict-driven repair: pick a conflicting difference, move one of its producers
            conf_list = [d for d, lst in dmap.items() if len(lst) > 1]
            if not conf_list:
                continue
            d_conf = rng.choice(conf_list)
            producers = dmap[d_conf]
            # Pick a random producer to move
            ri, ii, jj = rng.choice(producers)
            row = current_rows[ri]
            # Try to move element at position jj (or ii) to eliminate this difference
            # We'll try moving the later element (jj)
            pos = jj
            old_val = row[pos]
            prev_val = row[pos - 1]
            next_val = row[pos + 1] if pos < len(row) - 1 else current_scope + 20
            
            lo = prev_val + 1
            hi = next_val - 1
            if lo > hi:
                # Try the earlier element instead
                pos = ii
                old_val = row[pos]
                prev_val = row[pos - 1] if pos > 0 else -1
                next_val = row[pos + 1] if pos < len(row) - 1 else current_scope + 20
                lo = prev_val + 1 if pos > 0 else 0
                hi = next_val - 1
                if lo > hi or lo == 0:
                    continue
            
            # Try several candidate positions
            best_new = None
            best_new_cost = float('inf')
            candidates = set()
            span = hi - lo
            for _ in range(12):
                delta = int(rng.gauss(0, max(1, span * 0.25)))
                nv = old_val + delta
                if lo <= nv <= hi and nv != old_val:
                    candidates.add(nv)
            # Also try close to old value
            for off in range(1, 6):
                if old_val - off >= lo:
                    candidates.add(old_val - off)
                if old_val + off <= hi:
                    candidates.add(old_val + off)
            
            for nv in candidates:
                if nv == old_val:
                    continue
                # Apply
                row[pos] = nv
                new_dmap = _build_diff_map(current_rows)
                new_cost = _cost(current_rows, new_dmap)
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_new = nv
                    best_new_dmap = new_dmap
                row[pos] = old_val
            
            if best_new is None:
                continue
            
            # Accept with SA criterion
            row[pos] = best_new
            dmap = best_new_dmap
            new_cost = best_new_cost
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = _scope(current_rows)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                row[pos] = old_val
                dmap = _build_diff_map(current_rows)
                current_cost = _cost(current_rows, dmap)
        
        elif nc == 0:
            # No conflicts: try to reduce scope
            # Find the row with the maximum element and try to compress it
            max_val = current_scope
            max_ri = -1
            max_pos = -1
            for ri in range(n_rows):
                for pi in range(len(current_rows[ri]) - 1, 0, -1):
                    if current_rows[ri][pi] == max_val:
                        max_ri = ri
                        max_pos = pi
                        break
                if max_ri >= 0:
                    break
            
            if max_ri < 0:
                continue
            
            row = current_rows[max_ri]
            pos = max_pos
            old_val = row[pos]
            prev_val = row[pos - 1]
            lo = prev_val + 1
            hi = old_val - 1  # Must be strictly less to reduce scope
            
            if lo > hi:
                # Can't compress this element; try a random single-element move
                ri = rng.randint(0, n_rows - 1)
                pi = rng.randint(1, len(current_rows[ri]) - 1)
                row = current_rows[ri]
                pos = pi
                old_val = row[pos]
                prev_val = row[pos - 1]
                next_val = row[pos + 1] if pos < len(row) - 1 else current_scope + 20
                lo = prev_val + 1
                hi = next_val - 1
                if lo > hi:
                    continue
                nv = rng.randint(lo, hi)
                if nv == old_val:
                    continue
                row[pos] = nv
                new_dmap = _build_diff_map(current_rows)
                new_cost = _cost(current_rows, new_dmap)
                if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                    dmap = new_dmap
                    current_cost = new_cost
                    current_scope = _scope(current_rows)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = current_scope
                        best_rows = [list(r) for r in current_rows]
                else:
                    row[pos] = old_val
                    dmap = _build_diff_map(current_rows)
                    current_cost = _cost(current_rows, dmap)
                T = max(T * cooling, min_T)
                continue
            
            # Try to compress
            best_new = None
            best_new_cost = float('inf')
            # Prefer values close to lo (maximum compression)
            for nv in range(lo, min(lo + 20, hi + 1)):
                row[pos] = nv
                new_dmap = _build_diff_map(current_rows)
                new_cost = _cost(current_rows, new_dmap)
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_new = nv
                    best_new_dmap = new_dmap
                row[pos] = old_val
            
            # Also try some random values in range
            for _ in range(8):
                nv = rng.randint(lo, hi)
                if nv == old_val or (best_new is not None and nv == best_new):
                    continue
                row[pos] = nv
                new_dmap = _build_diff_map(current_rows)
                new_cost = _cost(current_rows, new_dmap)
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_new = nv
                    best_new_dmap = new_dmap
                row[pos] = old_val
            
            if best_new is None:
                continue
            
            row[pos] = best_new
            dmap = best_new_dmap
            new_cost = best_new_cost
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = _scope(current_rows)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                row[pos] = old_val
                dmap = _build_diff_map(current_rows)
                current_cost = _cost(current_rows, dmap)
        
        else:
            # Mixed: some conflicts, do a random single-element move
            ri = rng.randint(0, n_rows - 1)
            pi = rng.randint(1, len(current_rows[ri]) - 1)
            row = current_rows[ri]
            pos = pi
            old_val = row[pos]
            prev_val = row[pos - 1]
            next_val = row[pos + 1] if pos < len(row) - 1 else current_scope + 20
            lo = prev_val + 1
            hi = next_val - 1
            if lo > hi:
                continue
            nv = rng.randint(lo, hi)
            if nv == old_val:
                continue
            row[pos] = nv
            new_dmap = _build_diff_map(current_rows)
            new_cost = _cost(current_rows, new_dmap)
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                dmap = new_dmap
                current_cost = new_cost
                current_scope = _scope(current_rows)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                row[pos] = old_val
                dmap = _build_diff_map(current_rows)
                current_cost = _cost(current_rows, dmap)
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def _error_repair(rows, k, rng, deadline):
    """
    Error-repair strategy: start from a possibly invalid configuration,
    iteratively fix the most-conflicting element until valid, then optimize scope.
    This is a different search paradigm: repair-based vs. perturbation-based.
    """
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    
    # Phase 1: Repair - eliminate all conflicts
    max_repair_iters = 5000
    for it in range(max_repair_iters):
        if time.time() > deadline:
            break
        
        dmap = _build_diff_map(current_rows)
        nc = _n_conflicts(dmap)
        if nc == 0:
            break
        
        # Find the difference with the most conflicts
        worst_d = None
        worst_count = 0
        for d, lst in dmap.items():
            if len(lst) > worst_count:
                worst_count = len(lst)
                worst_d = d
        
        if worst_d is None:
            break
        
        # Pick one of the producers to move
        producers = dmap[worst_d]
        # Prefer to move the one with the largest value (easier to relocate)
        producers_sorted = sorted(producers, key=lambda x: current_rows[x[0]][x[2]], reverse=True)
        ri, ii, jj = producers_sorted[0]
        
        row = current_rows[ri]
        pos = jj
        old_val = row[pos]
        prev_val = row[pos - 1]
        next_val = row[pos + 1] if pos < len(row) - 1 else _scope(current_rows) + 50
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            pos = ii
            old_val = row[pos]
            prev_val = row[pos - 1] if pos > 0 else -1
            next_val = row[pos + 1] if pos < len(row) - 1 else _scope(current_rows) + 50
            lo = prev_val + 1 if pos > 0 else 1
            hi = next_val - 1
            if lo > hi:
                continue
        
        # Find a value that doesn't create new conflicts
        scope = _scope(current_rows)
        best_nv = None
        best_nc = nc  # Current number of conflicts
        
        # Try values in a range
        search_range = max(10, int(scope * 0.3))
        tried = set()
        for _ in range(50):
            nv = rng.randint(lo, min(hi, lo + search_range))
            if nv in tried or nv == old_val:
                continue
            tried.add(nv)
            row[pos] = nv
            new_dmap = _build_diff_map(current_rows)
            new_nc = _n_conflicts(new_dmap)
            if new_nc < best_nc:
                best_nc = new_nc
                best_nv = nv
                best_nv_dmap = new_dmap
            if new_nc == 0:
                break
            row[pos] = old_val
        
        if best_nv is None:
            # Try a wider random search
            for _ in range(30):
                nv = rng.randint(lo, hi)
                if nv in tried or nv == old_val:
                    continue
                tried.add(nv)
                row[pos] = nv
                new_dmap = _build_diff_map(current_rows)
                new_nc = _n_conflicts(new_dmap)
                if new_nc < best_nc:
                    best_nc = new_nc
                    best_nv = nv
                    best_nv_dmap = new_dmap
                if new_nc == 0:
                    break
                row[pos] = old_val
        
        if best_nv is not None:
            row[pos] = best_nv
            dmap = best_nv_dmap
        else:
            # Accept a random move to escape
            nv = rng.randint(lo, hi)
            if nv != old_val:
                row[pos] = nv
                dmap = _build_diff_map(current_rows)
    
    # Phase 2: Optimize scope on the (hopefully) valid configuration
    if time.time() > deadline:
        return _scope(current_rows), current_rows
    
    # Check if valid
    dmap = _build_diff_map(current_rows)
    if _n_conflicts(dmap) == 0:
        # Run a short SA to reduce scope
        remaining = deadline - time.time()
        iters = int(remaining * 6000)
        iters = max(iters, 1000)
        s, rows = _sa_optimize_conflict_driven(current_rows, k, iters, rng, deadline)
        return s, rows
    
    return _scope(current_rows), current_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 140.0  # Safety margin before 150s kill
    
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
    
    # Standard greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    # Spread greedy  
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Randomized greedy with different seeds
    for seed in [1, 7, 13, 23]:
        rng2 = random.Random(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng2.randint(1, 10)
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
    
    # Sort by scope
    candidates.sort(key=_scope)
    
    # Strategy 1: Conflict-driven SA on best candidates
    for idx, cand in enumerate(candidates[:3]):
        if time.time() > _DEADLINE - 30:
            break
        remaining = _DEADLINE - time.time() - 30
        if idx == 0:
            sa_deadline = time.time() + remaining * 0.4
        elif idx == 1:
            sa_deadline = time.time() + remaining * 0.3
        else:
            sa_deadline = _DEADLINE - 30
        iters = int((sa_deadline - time.time()) * 5000)
        iters = max(iters, 1000)
        s, rows = _sa_optimize_conflict_driven(cand, k, iters, rng, sa_deadline)
        consider(rows)
    
    # Strategy 2: Error repair on different candidates
    if time.time() < _DEADLINE - 20:
        # Use a perturbed version of a candidate
        if len(candidates) > 3:
            perturbed = [list(r) for r in candidates[3]]
            # Perturb some elements
            for _ in range(n_rows * k // 2):
                ri = rng.randint(0, n_rows - 1)
                pi = rng.randint(1, k)
                old = perturbed[ri][pi]
                lo = perturbed[ri][pi-1] + 1
                hi = perturbed[ri][pi+1] - 1 if pi < k else old + 10
                if lo < hi:
                    perturbed[ri][pi] = rng.randint(lo, hi)
            
            remaining = _DEADLINE - time.time() - 15
            s, rows = _error_repair(perturbed, k, rng, time.time() + remaining)
            consider(rows)
    
    # Final polish: if we have a valid best, do a short conflict-driven SA
    if _BEST is not None and time.time() < _DEADLINE - 5:
        remaining = _DEADLINE - time.time() - 5
        iters = int(remaining * 8000)
        if iters > 500:
            s, rows = _sa_optimize_conflict_driven(_BEST[1], k, iters, rng, _DEADLINE - 5)
            consider(rows)
    
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
