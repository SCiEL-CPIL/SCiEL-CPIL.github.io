import time
import random
import math

def _validate(rows):
    diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _scope(rows):
    return max(max(r) for r in rows)

def _greedy_fallback(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _fast_greedy(n_rows, k, rng, look_ahead=8):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            start = row[-1] + 1
            cands = []
            for cand in range(start, start + look_ahead + 1):
                nd = {cand - a for a in row}
                if not nd & used:
                    cands.append(cand)
            if not cands:
                cand = start
                while True:
                    nd = {cand - a for a in row}
                    if not nd & used:
                        cands.append(cand)
                        break
                    cand += 1
                    if cand > 600:
                        break
            if not cands:
                cand = start
                while True:
                    nd = {cand - a for a in row}
                    if not nd & used:
                        cands.append(cand)
                        break
                    cand += 1
            cands.sort()
            if rng.random() < 0.75 or len(cands) == 1:
                chosen = cands[0]
            else:
                chosen = cands[rng.randrange(min(3, len(cands)))]
            row.append(chosen)
            used |= {chosen - a for a in row[:-1]}
        rows.append(row)
    return rows

class _SAState:
    __slots__ = ['rows', 'diff_counts', 'scope', 'n_rows', 'k', 'total_diffs']
    
    def __init__(self, rows):
        self.rows = [list(r) for r in rows]
        self.n_rows = len(rows)
        self.k = len(rows[0]) - 1
        self.total_diffs = self.n_rows * (self.k + 1) * self.k // 2
        self.diff_counts = {}
        self.scope = 0
        for r in self.rows:
            self.scope = max(self.scope, r[-1])
            n = len(r)
            for i in range(n):
                for j in range(i + 1, n):
                    d = r[j] - r[i]
                    self.diff_counts[d] = self.diff_counts.get(d, 0) + 1
    
    def _remove_row(self, ri):
        r = self.rows[ri]
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                self.diff_counts[d] -= 1
                if self.diff_counts[d] == 0:
                    del self.diff_counts[d]
    
    def _add_row(self, ri):
        r = self.rows[ri]
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                self.diff_counts[d] = self.diff_counts.get(d, 0) + 1
    
    def _is_valid(self):
        return len(self.diff_counts) == self.total_diffs
    
    def _recompute_scope(self):
        self.scope = max(r[-1] for r in self.rows)

def _sa_optimize(rows, deadline, rng):
    state = _SAState(rows)
    best_rows = [list(r) for r in state.rows]
    best_scope = state.scope
    
    T = 50.0
    T_min = 1.0
    cooling = 0.9995
    
    t0 = time.time()
    while time.time() < deadline:
        if T < T_min:
            T = 15.0
            # Perturb
            ri = rng.randrange(state.n_rows)
            p = rng.randrange(1, len(state.rows[ri]))
            lo = state.rows[ri][p - 1] + 1
            hi = state.rows[ri][p] + 8
            new_val = rng.randint(lo, hi)
            old_val = state.rows[ri][p]
            state.rows[ri][p] = new_val
            state._remove_row(ri)
            state._add_row(ri)
            state._recompute_scope()
            if not state._is_valid():
                state.rows[ri][p] = old_val
                state._remove_row(ri)
                state._add_row(ri)
                state._recompute_scope()
            else:
                if state.scope < best_scope:
                    best_scope = state.scope
                    best_rows = [list(r) for r in state.rows]
            continue
        
        # Pick a move
        ri = rng.randrange(state.n_rows)
        p = rng.randrange(1, len(state.rows[ri]))
        delta = rng.choice([1, 1, 2, 2, 3, 3, 4, 5, 6, 8, 10])
        if rng.random() < 0.5:
            new_val = state.rows[ri][p] - delta
        else:
            new_val = state.rows[ri][p] + delta
        
        lo = state.rows[ri][p - 1] + 1
        if new_val <= lo or (p < len(state.rows[ri]) - 1 and new_val >= state.rows[ri][p + 1]):
            continue
        
        old_val = state.rows[ri][p]
        if new_val == old_val:
            continue
        
        state.rows[ri][p] = new_val
        state._remove_row(ri)
        state._add_row(ri)
        
        if not state._is_valid():
            state.rows[ri][p] = old_val
            state._remove_row(ri)
            state._add_row(ri)
            T *= cooling
            continue
        
        state._recompute_scope()
        dE = state.scope - best_scope
        
        if dE <= 0 or rng.random() < math.exp(-dE / T):
            if state.scope < best_scope:
                best_scope = state.scope
                best_rows = [list(r) for r in state.rows]
        else:
            state.rows[ri][p] = old_val
            state._remove_row(ri)
            state._add_row(ri)
            state._recompute_scope()
        
        T *= cooling
    
    return best_rows, best_scope

def _row_rebuild(row, other_diffs, deadline, rng):
    """Try to minimize max entry of row given other_diffs are taken."""
    n = len(row)
    cur_max = row[-1]
    
    # Try targets from cur_max-1 downward
    for target in range(cur_max - 1, 0, -1):
        if time.time() > deadline:
            break
        found = _dfs_build(n, target, other_diffs, deadline, rng)
        if found is not None:
            return found
    return list(row)

def _dfs_build(n, max_allowed, other_diffs, deadline, rng):
    row = [0]
    
    def _build(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        lo = row[-1] + 1
        hi = max_allowed
        cands = list(range(lo, hi + 1))
        if len(cands) > 20:
            rng.shuffle(cands)
            cands.sort()
            cands = cands[:20]
        for cand in cands:
            row.append(cand)
            nd = {cand - a for a in row[:-1]}
            if nd & other_diffs:
                row.pop()
                continue
            # Check internal uniqueness
            ok = True
            seen = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        ok = False
                        break
                    seen.add(d)
                if not ok:
                    break
            if not ok:
                row.pop()
                continue
            if _build(pos + 1):
                return True
            row.pop()
        return False
    
    if _build(1):
        return list(row)
    return None

def _row_reoptimize(rows, deadline, rng):
    best = [list(r) for r in rows]
    best_sc = _scope(best)
    n_rows = len(best)
    
    for _pass in range(5):
        if time.time() > deadline:
            break
        order = sorted(range(n_rows), key=lambda i: best[i][-1], reverse=True)
        improved = False
        for ri in order:
            if time.time() > deadline:
                break
            other_diffs = set()
            for rj in range(n_rows):
                if rj == ri:
                    continue
                r = best[rj]
                n = len(r)
                for i in range(n):
                    for j in range(i + 1, n):
                        other_diffs.add(r[j] - r[i])
            new_row = _row_rebuild(best[ri], other_diffs, deadline, rng)
            if new_row[-1] < best[ri][-1]:
                best[ri] = new_row
                improved = True
                best_sc = _scope(best)
    return best, best_sc

def construct(n_rows, k):
    t_start = time.time()
    deadline = t_start + 135.0
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy restarts
    best_rows = _greedy_fallback(n_rows, k)
    best_sc = _scope(best_rows)
    
    phase1_deadline = t_start + 15.0
    while time.time() < phase1_deadline:
        rows = _fast_greedy(n_rows, k, rng, look_ahead=rng.choice([5, 6, 8, 10]))
        sc = _scope(rows)
        if sc < best_sc:
            best_sc = sc
            best_rows = rows
    
    # Phase 2: Simulated annealing with incremental diff tracking
    phase2_deadline = t_start + 110.0
    best_rows, best_sc = _sa_optimize(best_rows, phase2_deadline, rng)
    
    # Phase 3: Row re-optimization
    phase3_deadline = t_start + 130.0
    best_rows, best_sc = _row_reoptimize(best_rows, phase3_deadline, rng)
    
    # Phase 4: Final SA polish
    if time.time() < deadline - 5:
        best_rows, best_sc = _sa_optimize(best_rows, deadline, rng)
    
    # Final validation
    if not _validate(best_rows):
        best_rows = _greedy_fallback(n_rows, k)
    
    return best_rows
