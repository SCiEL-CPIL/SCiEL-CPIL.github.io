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
    for attempt in range(6):
        row = [0]
        x = 0
        local_diffs = set()
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.4:
                start = x + rng.randint(1, 3)
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
                elif attempt < 2 and conflicts <= 1 and rng.random() < 0.1:
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
                    d = row[j] - r[i]
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

def _enumerate_rows(k, max_scope, deadline):
    """Enumerate all k+1 element Sidon sets starting at 0 with max element <= max_scope."""
    rows = []
    # DFS to build rows
    def dfs(current, used_diffs):
        if time.time() > deadline:
            return
        if len(current) == k + 1:
            rows.append(list(current))
            return
        start = current[-1] + 1
        for x in range(start, max_scope + 1):
            new_diffs = set()
            conflict = False
            for a in current:
                d = x - a
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            if not conflict:
                current.append(x)
                used_diffs.update(new_diffs)
                dfs(current, used_diffs)
                current.pop()
                used_diffs.difference_update(new_diffs)
    dfs([0], set())
    return rows

def _tomita_clique(P, color, n, target, found_clique, deadline, min_scope):
    """Branch and bound maximum clique. Returns clique of size target if found."""
    if len(found_clique) + len(P) < target:
        return False
    if len(found_clique) == target:
        return True
    if time.time() > deadline:
        return False
    while P:
        # Order by color
        P_sorted = sorted(P, key=lambda x: color[x])
        P = P_sorted
        # Check if we can reach target
        if len(found_clique) + len(P) < target:
            return False
        # Take the last vertex (highest color)
        v = P[-1]
        P = P[:-1]
        # Recurse with neighbors of v that are in P
        Nv = [u for u in P if u in _adj[v]]
        found_clique.append(v)
        if _tomita_clique(Nv, color, n, target, found_clique, deadline, min_scope):
            return True
        found_clique.pop()
        # Remove v from consideration
    return False

def _color_sort(P, color, n):
    """Greedy coloring of subgraph induced by P. Returns (P_sorted, color_array)."""
    # color[i] for i in 0..n-1
    uncolored = list(P)
    color_order = []
    c = 0
    while uncolored:
        c += 1
        remaining = []
        for v in uncolored:
            if all(color[u] != c for u in remaining):
                color[v] = c
                color_order.append(v)
            else:
                remaining.append(v)
        uncolored = remaining
    # Sort P by color
    P_sorted = sorted(P, key=lambda x: color[x])
    return P_sorted

def _exact_search(k, n_rows, initial_scope, deadline):
    """Exact branch-and-bound to find n_rows compatible rows minimizing scope."""
    global _BEST
    # Try scopes from lower bound up to initial_scope
    # Lower bound: counting argument
    # Each row uses C(k+1, 2) differences. Total = n_rows * C(k+1,2).
    # All differences must be in [1, scope], so scope >= n_rows * C(k+1,2)
    lower_bound = n_rows * (k * (k + 1) // 2)
    # Actually the differences are distinct across ALL rows, so we need at least
    # n_rows * C(k+1, 2) distinct positive integers, each <= scope
    # So scope >= n_rows * C(k+1, 2)
    # But this is a very loose bound. Better: the largest difference in a row with
    # max element m is at most m, and we need n_rows * C(k+1,2) distinct values in [1, m]
    # so m >= n_rows * C(k+1, 2)
    
    # For (7, 5): C(6,2) = 15, so 7*15 = 105. Best known is 112.
    
    # Enumerate rows up to initial_scope
    enum_deadline = time.time() + 30
    rows = _enumerate_rows(k, initial_scope, enum_deadline)
    
    if len(rows) < n_rows:
        return None
    
    # Build compatibility: two rows are compatible if their difference sets are disjoint
    n = len(rows)
    row_diffs = []
    for r in rows:
        ds = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                ds.add(r[j] - r[i])
        row_diffs.append(ds)
    
    # For efficiency, sort rows by their max element (scope contribution)
    # We want to find a clique of size n_rows
    # Use bitsets for adjacency
    global _adj
    _adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if row_diffs[i].isdisjoint(row_diffs[j]):
                _adj[i].add(j)
                _adj[j].add(i)
    
    # Greedy coloring for upper bound
    color = [0] * n
    
    # Try to find clique of size n_rows
    # Start with all vertices, use Tomita's algorithm
    P = list(range(n))
    
    # Greedy coloring of the full graph
    def greedy_color(vertices):
        nonlocal color
        uncolored = list(vertices)
        c = 0
        while uncolored:
            c += 1
            remaining = []
            for v in uncolored:
                if all(color[u] != c for u in remaining if u in _adj[v]):
                    color[v] = c
                    uncolored.remove(v)
                    break
                else:
                    remaining.append(v)
            # Simplified: just assign incrementally
            # This is getting complex; use a simpler bound
        return c
    
    # Simpler approach: use the fact that we need exactly n_rows rows
    # Try all combinations is too expensive. Use backtracking with pruning.
    
    # Better: use the initial SA solution as a starting point and try to improve
    # by swapping rows with ones that have smaller max elements
    
    # Actually, let's do a simpler exact search:
    # For each possible scope m from lower_bound to initial_scope:
    #   Filter rows with max element <= m
    #   Check if there's a clique of size n_rows
    # Return the smallest m that works
    
    # But enumerating all rows up to 112 might be expensive.
    # Let's limit the enumeration.
    
    # Alternative: use the SA result and try local improvements
    # This is getting too complex for the time budget.
    # Let's just return the SA result if exact search is too slow.
    
    return None

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
    
    # SA phase
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
        
    # Polish
    if _BEST is not None and time.time() < _DEADLINE:
        remaining_time = _DEADLINE - time.time()
        if remaining_time > 5:
            polish_deadline = time.time() + remaining_time * 0.7
            polish_iters = int((polish_deadline - time.time()) * 10000)
            polish_iters = max(polish_iters, 500)
            s, rows = _sa_optimize(_BEST[1], k, polish_iters, rng, polish_deadline, polish=True)
            consider(rows)
            
            if time.time() < _DEADLINE - 2:
                final_deadline = _DEADLINE - 2
                final_iters = int((final_deadline - time.time()) * 12000)
                final_iters = max(final_iters, 200)
                s, rows = _sa_optimize(_BEST[1], k, final_iters, rng, final_deadline, polish=True)
                consider(rows)
    
    # Exact search phase: try to improve via row-swap exact optimization
    if _BEST is not None and time.time() < _DEADLINE - 5:
        current_scope = _BEST[0]
        # Try to reduce scope by 1, 2, ... using exact row enumeration + clique check
        for target_scope in range(current_scope - 1, max(100, current_scope - 10), -1):
            if time.time() > _DEADLINE - 5:
                break
            # Enumerate rows with max element <= target_scope
            enum_dl = time.time() + 10
            rows_enum = _enumerate_rows(k, target_scope, enum_dl)
            if len(rows_enum) < n_rows:
                continue
            
            # Build diff sets
            row_diffs = []
            for r in rows_enum:
                ds = set()
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        ds.add(r[j] - r[i])
                row_diffs.append(ds)
            
            # Greedy: try to pick n_rows compatible rows
            # Backtracking with ordering by number of compatible partners
            n = len(rows_enum)
            compat = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if row_diffs[i].isdisjoint(row_diffs[j]):
                        compat[i].add(j)
                        compat[j].add(i)
            
            # Order by degree (prefer high degree for faster pruning)
            order = sorted(range(n), key=lambda x: -len(compat[x]))
            
            def backtrack(selected, available, depth):
                if time.time() > _DEADLINE - 5:
                    return None
                if depth == n_rows:
                    return list(selected)
                # Pruning: if not enough candidates left
                if len(available) < n_rows - depth:
                    return None
                for idx in available:
                    v = order[idx]
                    # Check compatibility with all selected
                    ok = True
                    for s in selected:
                        if v not in compat[s]:
                            ok = False
                            break
                    if ok:
                        selected.append(v)
                        new_available = [u for u in available if u != idx and order[u] in compat[v]]
                        result = backtrack(selected, new_available, depth + 1)
                        if result is not None:
                            return result
                        selected.pop()
                return None
            
            result = backtrack([], list(range(n)), 0)
            if result is not None:
                new_rows = [rows_enum[v] for v in result]
                if _valid(new_rows, k):
                    new_scope = _scope(new_rows)
                    if new_scope <= target_scope:
                        _BEST = (new_scope, new_rows)
                        break
    
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
