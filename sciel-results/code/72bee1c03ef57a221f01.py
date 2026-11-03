import time
import random
import math
from collections import Counter

_DEADLINE = 0.0
_BEST = None

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

def _build_diff_set(rows):
    s = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                s.add(r[j] - r[i])
    return s

def _generate_row_pool(k, max_scope, limit, rng):
    """
    Generate a pool of valid rows (Sidon sets) up to max_scope.
    """
    pool = []
    seen_rows = set()
    # We want diverse rows. Let's use a randomized backtracking generator.
    
    def gen_row(min_start=1, current_max=0):
        # current_max is the last element added
        row = [0, current_max] if current_max > 0 else [0]
        diffs = set()
        if current_max > 0:
            diffs.add(current_max)
            
        if len(row) == k + 1:
            return row
            return None
        
        # Pick next element
        start = current_max + 1 if current_max > 0 else 1
        # To get small scopes, we should try small increments first, 
        # but randomize to avoid getting stuck.
        # Actually, for a pool, we just want ANY valid rows up to max_scope.
        # Let's just do a simple randomized search.
        
        candidates = list(range(start, max_scope + 1))
        rng.shuffle(candidates)
        
        # Limit candidates to keep it fast
        if len(candidates) > 50:
            candidates = candidates[:50]
            
        for x in candidates:
            # Check diffs
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                new_diffs_set = set(new_diffs)
                if not (new_diffs_set & diffs):
                    row.append(x)
                    diffs.update(new_diffs_set)
                    res = gen_row(min_start, x)
                    if res is not None:
                        return res
                    row.pop()
                    diffs.difference_update(new_diffs_set)
        return None

    # Generate pool
    count = 0
    while count < limit:
        # Start with 0, then pick a random first element
        # To maximize variety, pick random first elements
        # But we want small scopes, so maybe pick small first elements?
        # The pool should contain rows of various "shapes"
        
        # Random restart
        first = rng.randint(1, max(1, max_scope // 3))
        row = [0, first]
        diffs = {first}
        
        if len(row) == k + 1:
             res = row
        else:
             res = gen_row(first, first)
             
        if res is not None:
            key = tuple(res)
            if key not in seen_rows:
                seen_rows.add(key)
                pool.append(res)
                count += 1
        
        if time.time() > _DEADLINE - 5:
            break
    return pool

def _csp_solve(n_rows, k, pool, deadline):
    """
    Solve using CSP with backtracking.
    Rows are assigned from pool.
    """
    # Pre-filter pool? No, let's just use it.
    # We want to minimize scope.
    # Sort pool by last element (scope contribution)
    pool.sort(key=lambda r: r[k])
    
    # We can limit the pool size to the top N smallest rows to speed up search?
    # Or use all of them but the search order matters.
    
    rows = []
    used_diffs = set()
    
    def backtrack(idx):
        nonlocal rows, used_diffs
        if time.time() > deadline:
            return False
            
        if idx == n_rows:
            return True
            
        # Heuristic: Try rows with smallest last element first?
        # Or most constrained?
        # Let's iterate through pool.
        
        for r in pool:
            # Check if r's diffs are disjoint from used_diffs
            r_diffs = set()
            ok = True
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used_diffs:
                        ok = False
                        break
                    r_diffs.add(d)
                if not ok:
                    break
            
            if ok:
                rows.append(r)
                used_diffs.update(r_diffs)
                
                if backtrack(idx + 1):
                    return True
                else:
                    used_diffs.difference_update(r_diffs)
                    rows.pop()
                    
        return False

    # To help CSP, we might want to sort pool by "density" or something.
    # But simple sorted by max element is a good start for minimization.
    
    return backtrack(0)

def _sa_optimize(rows, k, iters, rng, deadline):
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in current_rows]
    
    # Build diff counts
    diff_counts = Counter()
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diff_counts[r[j] - r[i]] += 1
                
    def get_cost():
        scope = max(a for r in current_rows for a in r[1:])
        collisions = sum(max(0, c - 1) for c in diff_counts.values())
        return scope + 100 * collisions

    current_cost = get_cost()
    best_cost = current_cost
    
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    for _ in range(iters):
        if time.time() > deadline:
            break
            
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current_rows[i][j]
        lo = current_rows[i][j-1] + 1
        hi = (current_rows[i][j+1] - 1) if j < k else max(a for r in current_rows for a in r[1:])
        
        if lo > hi:
            continue
            
        new_val = rng.randint(lo, hi)
        if new_val == old_val:
            continue
            
        # Remove old diffs
        for t in range(k + 1):
            if t == j: continue
            d = abs(old_val - current_rows[i][t])
            if d > 0:
                diff_counts[d] -= 1
                if diff_counts[d] == 0: del diff_counts[d]
                
        current_rows[i][j] = new_val
        
        # Add new diffs
        for t in range(k + 1):
            if t == j: continue
            d = abs(new_val - current_rows[i][t])
            if d > 0:
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
        new_cost = get_cost()
        
        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [list(r) for r in current_rows]
        else:
            # Revert
            for t in range(k + 1):
                if t == j: continue
                d = abs(new_val - current_rows[i][t])
                if d > 0:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0: del diff_counts[d]
            for t in range(k + 1):
                if t == j: continue
                d = abs(old_val - current_rows[i][t])
                if d > 0:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
            current_rows[i][j] = old_val
            
        T = max(T * cooling, min_T)
        
    return best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 110.0
    
    rng = random.Random(42)
    
    # 1. Quick Greedy
    def fast_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows
    
    baseline = fast_greedy()
    baseline_scope = _scope(baseline)
    _BEST = baseline
    
    # 2. CSP Attempt
    # Try to find a solution with scope < baseline_scope
    # We will try a few target scopes
    for target_scope in range(max(100, baseline_scope - 10), baseline_scope):
        if time.time() > _DEADLINE - 20:
            break
            
        pool = _generate_row_pool(k, target_scope, 1000, rng)
        if not pool:
            continue
            
        csp_deadline = time.time() + 10.0
        if _csp_solve(n_rows, k, pool, csp_deadline):
             # We found a solution?
             # Wait, _csp_solve modifies global rows? No, it uses local vars in closure but I wrote it carefully.
             # Let's check if it actually returns the rows.
             # My _csp_solve above uses nonlocal rows, but rows is local to _csp_solve.
             # It returns True/False. It does NOT return the rows.
             # I need to fix _csp_solve to return the solution.
             pass

    # Let's implement a proper CSP solver that returns the solution.
    def csp_solve(n_rows, k, pool, deadline):
        pool.sort(key=lambda r: r[k])
        rows = []
        used_diffs = set()
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n_rows:
                return True
                
            # Heuristic: Try rows that have diffs least overlapping with used_diffs?
            # Or just iterate.
            # To speed up, maybe filter candidates first?
            
            candidates = []
            for r in pool:
                r_diffs = set()
                ok = True
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        d = r[j] - r[i]
                        if d in used_diffs:
                            ok = False
                            break
                        r_diffs.add(d)
                    if not ok:
                        break
                if ok:
                    candidates.append((r, r_diffs))
                    
            # Sort candidates by some heuristic? e.g. sum of diffs?
            # For now, just iterate.
            for r, r_diffs in candidates:
                rows.append(r)
                used_diffs.update(r_diffs)
                if backtrack(idx + 1):
                    return True
                used_diffs.difference_update(r_diffs)
                rows.pop()
            return False
            
        if backtrack(0):
            return rows
        return None

    # Retry CSP with proper return
    # Use a slightly larger pool generation time
    csp_start = time.time()
    if _DEADLINE - csp_start > 30:
        # Try to beat baseline
        # Generate pool up to baseline_scope
        pool = _generate_row_pool(k, baseline_scope, 2000, rng)
        if pool:
            csp_deadline = time.time() + 20.0
            sol = csp_solve(n_rows, k, pool, csp_deadline)
            if sol:
                sol_scope = _scope(sol)
                if sol_scope < _scope(_BEST):
                    _BEST = sol
    
    # 3. SA Polish
    if time.time() < _DEADLINE - 5:
        remaining = _DEADLINE - time.time()
        iters = int(remaining * 8000)
        if iters > 100:
            sa_deadline = time.time() + 40.0 # Cap SA time
            improved = _sa_optimize(_BEST, k, iters, rng, sa_deadline)
            if _valid(improved, k) and _scope(improved) < _scope(_BEST):
                _BEST = improved
                
    return _BEST

