import time
import random
import math
import numpy as np
from collections import Counter

_DEADLINE = 0.0
_BEST = None

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

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _fast_greedy(k, n_rows, rng, spread=False):
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

def _gen_random_row(max_scope, k, rng, existing_diffs=None):
    """Generate a random row with scope <= max_scope, avoiding existing_diffs if provided."""
    row = [0]
    x = 0
    row_diffs = set()
    
    if existing_diffs is None:
        existing_diffs = set()
    
    for _ in range(k):
        # Try to pick next element
        found = False
        # Randomized search: try random candidates first, then sequential
        candidates = list(range(x + 1, min(max_scope, x + 50) + 1))
        rng.shuffle(candidates)
        
        for t in candidates:
            if t > max_scope:
                continue
            ok = True
            new_diffs = []
            for a in row:
                d = t - a
                if d in row_diffs or d in existing_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                row.append(t)
                row_diffs.update(new_diffs)
                x = t
                found = True
                break
        
        if not found:
            # Try sequential scan
            for t in range(x + 1, max_scope + 1):
                ok = True
                new_diffs = []
                for a in row:
                    d = t - a
                    if d in row_diffs or d in existing_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                if ok:
                    row.append(t)
                    row_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
        
        if not found:
            return None
    
    return row

def _sa_vectorized(rows, k, deadline, rng, T_init=10.0, cooling=0.9997, min_T=0.05):
    n_rows = len(rows)
    mat = np.array(rows, dtype=np.int32)
    current_scope = int(mat.max())
    
    n_pairs = (k + 1) * k // 2
    row_idx = []
    col_idx = []
    for i in range(k + 1):
        for j in range(i + 1, k + 1):
            row_idx.append(i)
            col_idx.append(j)
    row_idx = np.array(row_idx, dtype=np.int32)
    col_idx = np.array(col_idx, dtype=np.int32)
    
    def get_all_diffs(m):
        diffs_per_row = m[:, col_idx] - m[:, row_idx]
        return diffs_per_row.flatten()
    
    def compute_cost(m):
        diffs = get_all_diffs(m)
        if len(diffs) == 0:
            return int(m.max()), 0
        unique, counts = np.unique(diffs, return_counts=True)
        collisions = int(np.sum(np.maximum(0, counts - 1)))
        scope = int(m.max())
        return scope, collisions
    
    current_scope, current_collisions = compute_cost(mat)
    W = 15.0
    current_cost = current_scope + W * current_collisions
    best_cost = current_cost
    best_mat = mat.copy()
    best_scope = current_scope
    
    T = T_init
    n_iters = 0
    stagnant = 0
    MAX_STAGNANT = 3000
    
    while time.time() < deadline:
        n_iters += 1
        r = rng.random()
        
        if r < 0.25:
            # Row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row = mat[i].copy()
            
            other_mat = np.delete(mat, i, axis=0)
            other_diffs = get_all_diffs(other_mat)
            used_diffs = set(int(d) for d in other_diffs)
            
            new_row = [0]
            x = 0
            ok = True
            for j in range(1, k + 1):
                found = False
                for attempt in range(50):
                    if attempt == 0:
                        start = x + 1
                        end = x + 1 + max(1, (k - j + 1) * 3)
                    else:
                        start = x + 1
                        end = x + 1 + attempt * 5
                    for t in range(start, min(end, current_scope + 10)):
                        conflict = False
                        for a in new_row:
                            d = t - a
                            if d <= 0:
                                continue
                            if d in used_diffs:
                                conflict = True
                                break
                        if not conflict:
                            new_row.append(t)
                            for a in new_row[:-1]:
                                used_diffs.add(t - a)
                            x = t
                            found = True
                            break
                    if found:
                        break
                if not found:
                    ok = False
                    break
            
            if not ok or len(new_row) != k + 1:
                continue
            
            new_row_arr = np.array(new_row, dtype=np.int32)
            if new_row_arr.max() > current_scope + 5:
                continue
            
            mat[i] = new_row_arr
            new_scope, new_collisions = compute_cost(mat)
            new_cost = new_scope + W * new_collisions
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_collisions = new_collisions
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
                    best_scope = new_scope
                    stagnant = 0
            else:
                mat[i] = old_row
                
        elif r < 0.45:
            # Row shift
            i = rng.randint(0, n_rows - 1)
            shift = rng.randint(-3, 3)
            if shift == 0:
                continue
            old_row = mat[i].copy()
            new_row = old_row.copy()
            new_row[1:] += shift
            if new_row[1:].min() <= 0 or new_row[1:].max() > current_scope + 10:
                continue
            if np.any(np.diff(new_row) <= 0):
                continue
            mat[i] = new_row
            new_scope, new_collisions = compute_cost(mat)
            new_cost = new_scope + W * new_collisions
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_collisions = new_collisions
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
                    best_scope = new_scope
                    stagnant = 0
            else:
                mat[i] = old_row
                
        elif r < 0.70:
            # Single-element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            old_val = int(mat[i, j])
            prev_val = int(mat[i, j - 1])
            next_val = int(mat[i, j + 1]) if j < k else current_scope + 100
            
            lo = prev_val + 1
            hi = next_val - 1
            if lo > hi:
                continue
            
            if old_val == current_scope:
                new_val = rng.randint(lo, min(hi, old_val - 1))
            else:
                span = hi - lo
                delta = int(rng.gauss(0, max(1, span * 0.25)))
                new_val = old_val + delta
            
            if new_val == old_val or new_val < lo or new_val > hi:
                continue
            
            mat[i, j] = new_val
            new_scope, new_collisions = compute_cost(mat)
            new_cost = new_scope + W * new_collisions
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_collisions = new_collisions
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
                    best_scope = new_scope
                    stagnant = 0
            else:
                mat[i, j] = old_val
                
        else:
            # Pairwise element move
            i = rng.randint(0, n_rows - 1)
            j1, j2 = rng.sample(range(1, k + 1), 2)
            if j1 > j2:
                j1, j2 = j2, j1
            
            old_v1 = int(mat[i, j1])
            old_v2 = int(mat[i, j2])
            
            prev1 = int(mat[i, j1 - 1])
            next1 = int(mat[i, j1 + 1]) if j1 < k else current_scope + 100
            prev2 = int(mat[i, j2 - 1])
            next2 = int(mat[i, j2 + 1]) if j2 < k else current_scope + 100
            
            lo1 = prev1 + 1
            hi1 = next1 - 1
            lo2 = prev2 + 1
            hi2 = next2 - 1
            
            if lo1 > hi1 or lo2 > hi2:
                continue
            
            new_v1 = rng.randint(lo1, hi1)
            new_v2 = rng.randint(lo2, hi2)
            
            if new_v1 == old_v1 and new_v2 == old_v2:
                continue
            
            mat[i, j1] = new_v1
            mat[i, j2] = new_v2
            if np.any(np.diff(mat[i]) <= 0):
                mat[i, j1] = old_v1
                mat[i, j2] = old_v2
                continue
            
            new_scope, new_collisions = compute_cost(mat)
            new_cost = new_scope + W * new_collisions
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_collisions = new_collisions
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
                    best_scope = new_scope
                    stagnant = 0
            else:
                mat[i, j1] = old_v1
                mat[i, j2] = old_v2
        
        T = max(T * cooling, min_T)
        
        if n_iters % 500 == 0 and stagnant > MAX_STAGNANT:
            mat = best_mat.copy()
            current_scope = best_scope
            current_scope, current_collisions = compute_cost(mat)
            current_cost = current_scope + W * current_collisions
            for _ in range(rng.randint(1, 3)):
                i = rng.randint(0, n_rows - 1)
                j = rng.randint(1, k)
                prev_val = int(mat[i, j - 1])
                next_val = int(mat[i, j + 1]) if j < k else current_scope + 100
                lo = prev_val + 1
                hi = next_val - 1
                if lo <= hi:
                    mat[i, j] = rng.randint(lo, hi)
                    if np.any(np.diff(mat[i]) <= 0):
                        mat[i, j] = prev_val + 1
            current_scope, current_collisions = compute_cost(mat)
            current_cost = current_scope + W * current_collisions
            T = T_init * 0.5
            stagnant = 0
    
    return best_scope, [list(map(int, r)) for r in best_mat]

def _dfs_row_minimize(row, k, used_diffs, max_scope, deadline):
    best_row = None
    best_max = max_scope + 1
    used_set = set(used_diffs)
    
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
        for t in range(current_x + 1, best_max):
            ok = True
            for a in current_row:
                d = t - a
                if d <= 0:
                    continue
                if d in used_set:
                    ok = False
                    break
            if ok:
                current_row.append(t)
                for a in current_row[:-1]:
                    used_set.add(t - a)
                dfs(pos + 1, current_row, t)
                for a in current_row[:-1]:
                    used_set.discard(t - a)
                current_row.pop()
            if time.time() > deadline:
                return
    
    dfs(1, [0], 0)
    return best_row

def _knapsack_packing(n_rows, k, deadline, rng):
    greedy_rows = _fast_greedy(k, n_rows, rng, spread=False)
    M_initial = max(r[-1] for r in greedy_rows)
    M_max = M_initial + 20
    
    def gen_row_random(max_scope, rng):
        row = [0]
        x = 0
        row_diffs = set()
        
        for _ in range(k):
            valid_vals = []
            for t in range(x + 1, max_scope + 1):
                ok = True
                for a in row:
                    d = t - a
                    if d in row_diffs:
                        ok = False
                        break
                if ok:
                    valid_vals.append(t)
            
            if not valid_vals:
                return None
            
            t = rng.choice(valid_vals)
            row.append(t)
            x = t
            for a in row[:-1]:
                d = t - a
                row_diffs.add(d)
        
        return row
    
    N_CANDIDATES = 5000
    candidates = []
    for _ in range(N_CANDIDATES):
        if time.time() > deadline - 10:
            break
        row = gen_row_random(M_max, rng)
        if row:
            bitset = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    bitset |= (1 << d)
            scope = row[-1]
            candidates.append((scope, bitset, row))
    
    if not candidates:
        return None, None
    
    candidates.sort(key=lambda x: x[0])
    
    def greedy_select(cands, n_needed):
        used = 0
        selected = []
        max_scope = 0
        for scope, bitset, row in cands:
            if (used & bitset) == 0:
                selected.append(row)
                used |= bitset
                max_scope = max(max_scope, scope)
                if len(selected) == n_needed:
                    return max_scope, selected
        return None, None
    
    best_scope, best_rows = greedy_select(candidates, n_rows)
    
    if best_scope is None:
        return None, None
    
    return best_scope, best_rows

def _branch_and_bound_search(n_rows, k, deadline, rng, initial_rows=None):
    """
    Branch and bound search treating the problem as finding n_rows disjoint
    difference sets (Sidon-like) with minimal maximum scope.
    """
    if initial_rows is None:
        initial_rows = _fast_greedy(k, n_rows, rng)
    
    current_scope = _scope(initial_rows)
    best_scope = current_scope
    best_rows = [list(r) for r in initial_rows]
    
    # Precompute all possible rows up to current_scope
    # This is expensive, so we'll use a smarter approach
    # Instead, we'll do iterative deepening on scope
    
    # For each row, compute its difference set as a frozenset
    def get_diff_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return frozenset(diffs)
    
    current_diff_sets = [get_diff_set(r) for r in best_rows]
    current_union = set()
    for ds in current_diff_sets:
        current_union.update(ds)
    
    # B&B: try to reduce scope by replacing rows
    # We'll use a priority queue of states
    # State: (max_scope, rows, diff_union)
    # We start from the current solution and try to improve
    
    # Actually, for this problem size, a simpler approach works:
    # Try to find n_rows disjoint difference sets with scope <= M
    # for M from lower_bound to current_scope
    
    # Lower bound: counting argument
    # Each row uses k*(k+1)/2 differences
    # Total differences needed: n_rows * k*(k+1)/2
    # Max difference is M, so we need n_rows * k*(k+1)/2 <= M
    # Thus M >= n_rows * k*(k+1) / 2
    
    lower_bound = max(10, (n_rows * k * (k + 1)) // 2)
    
    # Try scopes from lower_bound to current_scope
    # For each scope M, try to find n_rows disjoint rows with scope <= M
    
    def find_disjoint_rows(M, n_needed, deadline):
        """Find n_needed rows with scope <= M and disjoint difference sets."""
        # Generate candidate rows with scope <= M
        # Use randomized generation
        candidates = []
        seen_patterns = set()
        
        max_candidates = min(2000, max(100, n_needed * 50))
        
        for _ in range(max_candidates):
            if time.time() > deadline - 5:
                break
            row = _gen_random_row(M, k, rng)
            if row is None:
                continue
            diff_set = frozenset(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))
            pattern = tuple(sorted(diff_set))
            if pattern not in seen_patterns:
                seen_patterns.add(pattern)
                candidates.append((M, diff_set, row))
        
        if not candidates:
            return None
        
        # Sort by scope (all <= M, so just randomize)
        rng.shuffle(candidates)
        
        # Greedy + local search
        best = None
        best_size = 0
        
        for attempt in range(20):
            if time.time() > deadline - 3:
                break
            
            used_diffs = set()
            selected = []
            
            # Greedy: pick rows that conflict least
            remaining = candidates[:]
            rng.shuffle(remaining)
            
            for scope, diff_set, row in remaining:
                if len(selected) == n_needed:
                    break
                if not (diff_set & used_diffs):
                    selected.append(row)
                    used_diffs.update(diff_set)
            
            if len(selected) == n_needed and best is None:
                best = selected
                best_size = n_needed
            
            # Local search: try to swap rows to improve
            # (simplified: just try a few random replacements)
            for _ in range(5):
                if time.time() > deadline - 3:
                    break
                if not selected:
                    break
                
                i = rng.randint(0, len(selected) - 1)
                old_row = selected[i]
                old_diffs = frozenset(old_row[j] - old_row[i] for i in range(len(old_row)) for j in range(i + 1, len(old_row)))
                
                # Find a replacement
                for scope, diff_set, row in candidates:
                    if row == old_row:
                        continue
                    # Check if we can swap
                    used_without_i = set()
                    for r in selected:
                        if r != old_row:
                            used_without_i.update(r[j] - r[i] for i in range(len(r)) for j in range(i + 1, len(r)))
                    
                    if not (diff_set & used_without_i):
                        selected[i] = row
                        break
        
        return best
    
    # Try increasing M from lower_bound
    for M in range(lower_bound, current_scope + 1):
        if time.time() > deadline - 5:
            break
        
        rows = find_disjoint_rows(M, n_rows, deadline)
        if rows is not None:
            if _valid(rows, k):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in rows]
                break
    
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

    # 1. Try the Knapsack/Packing strategy
    knap_scope, knap_rows = _knapsack_packing(n_rows, k, _DEADLINE, rng)
    if knap_rows:
        consider(knap_rows)

    # 2. Try Branch and Bound
    if time.time() < _DEADLINE - 20:
        bb_scope, bb_rows = _branch_and_bound_search(n_rows, k, _DEADLINE - 10, rng)
        if bb_rows:
            consider(bb_rows)

    # 3. Generate diverse initial candidates
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    for _ in range(3):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 10)
                for t in range(x + 1, x + 1 + max_step * 4):
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
    
    if knap_rows:
        candidates.append(knap_rows)

    candidates.sort(key=_scope)

    # Run vectorized SA on top candidates
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
        s, rows = _sa_vectorized(cand, k, sa_deadline, rng)
        consider(rows)

    # DFS refinement on best result
    if _BEST is not None and time.time() < _DEADLINE - 5:
        best_rows = [list(r) for r in _BEST[1]]
        all_diffs = set()
        for r in best_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    all_diffs.add(r[j] - r[i])
        current_scope = _scope(best_rows)
        improved = True
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            for i in range(n_rows):
                if time.time() > _DEADLINE - 5:
                    break
                old_row = list(best_rows[i])
                row_diffs = set()
                for a in range(len(old_row)):
                    for b in range(a + 1, len(old_row)):
                        row_diffs.add(old_row[b] - old_row[a])
                other_diffs = all_diffs - row_diffs
                new_row = _dfs_row_minimize(old_row, k, other_diffs, current_scope, _DEADLINE - 5)
                if new_row is not None and new_row[-1] < old_row[-1]:
                    best_rows[i] = new_row
                    all_diffs = other_diffs | set(
                        new_row[b] - new_row[a]
                        for a in range(len(new_row))
                        for b in range(a + 1, len(new_row))
                    )
                    current_scope = min(current_scope, _scope(best_rows))
                    improved = True
                else:
                    all_diffs = other_diffs | row_diffs
                    best_rows[i] = old_row
        consider(best_rows)

    # Final polish SA
    if _BEST is not None and time.time() < _DEADLINE - 3:
        remaining = _DEADLINE - time.time()
        if remaining > 3:
            s, rows = _sa_vectorized(_BEST[1], k, _DEADLINE - 1, rng, T_init=3.0, cooling=0.9999, min_T=0.01)
            consider(rows)

    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
