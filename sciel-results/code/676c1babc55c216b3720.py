import time
import random
import numpy as np

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

def _diffs(row):
    return [row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))]

def _greedy_build(n_rows, k, rng, tight=True):
    """Fast greedy construction to give SA a valid starting point."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Try a few random candidates before falling back to linear scan
            best_x = None
            best_diffs = None
            # Random tries
            for _t in range(20):
                candidate = x + 1 + rng.randint(0, 40)
                ok = True
                new_diffs = []
                for a in row:
                    d = candidate - a
                    if d <= 0 or d in used:
                        ok = False
                        break
                    new_diffs.append(d)
                if ok:
                    best_x = candidate
                    best_diffs = new_diffs
                    break
            if best_x is None:
                # Linear scan
                for candidate in range(x + 1, x + 200):
                    ok = True
                    new_diffs = []
                    for a in row:
                        d = candidate - a
                        if d in used:
                            ok = False
                            break
                            new_diffs.append(d)
                    if ok:
                        best_x = candidate
                        best_diffs = [candidate - a for a in row]
                        break
            if best_x is None:
                best_x = x + 200
                best_diffs = [best_x - a for a in row]
            row.append(best_x)
            x = best_x
            for d in best_diffs:
                used.add(d)
        rows.append(row)
    return rows

def _sa_optimize(n_rows, k, initial_rows, deadline, rng):
    """
    Simulated annealing in difference-space.
    Representation: each row is a list of k+1 integers starting at 0.
    We maintain:
      - used_mask: numpy bool array of size max_scope+1 indicating used differences
      - row_diffs: list of lists, row_diffs[i] = list of all pairwise diffs in row i
    Move: pick a random row, pick a random position j (1..k), pick a random
          new difference d (unused), set row[j] += d? No -- that's not valid.
    
    Actually, a cleaner move in difference-space:
    Each row has k "steps": s_i = row[i] - row[i-1] for i=1..k.
    The row is determined by cumulative sums of steps.
    A move: pick a row, pick a step position p (1..k), change step s_p to s_p'
    such that the new row remains strictly increasing (s_p' >= 1) and all
    pairwise differences remain distinct and unused.
    
    But checking all pairwise differences after changing one step is O(k^2).
    For k=6, that's 21 diffs per row -- very cheap.
    
    Let me use step-based representation.
    """
    # Convert rows to steps
    steps = []
    for r in initial_rows:
        st = [r[i] - r[i-1] for i in range(1, k+1)]
        steps.append(st)
    
    # Build used difference set
    used = set()
    for r in initial_rows:
        for d in _diffs(r):
            used.add(d)
    
    best_scope = _scope(initial_rows)
    best_steps = [list(s) for s in steps]
    cur_steps = [list(s) for s in steps]
    
    # Temperature schedule
    t0 = 50.0
    t_min = 0.1
    # We'll run until deadline
    start_time = time.time()
    
    # Precompute: for a given row (list of k+1 values starting at 0), 
    # get all pairwise diffs
    def _row_values(st):
        vals = [0]
        s = 0
        for v in st:
            s += v
            vals.append(s)
        return vals
    
    def _row_diffs_from_values(vals):
        return [vals[j] - vals[i] for i in range(len(vals)) for j in range(i+1, len(vals))]
    
    # Initial state
    cur_vals = [_row_values(s) for s in cur_steps]
    cur_scope = _scope(cur_vals)
    
    n_iter = 0
    while time.time() < deadline:
        # Adaptive temperature
        elapsed = time.time() - start_time
        # Linear cooling
        frac = min(1.0, elapsed / max(1.0, deadline - start_time))
        T = t0 * (t_min / t0) ** frac
        
        # Pick random row and step
        ri = rng.randint(0, n_rows - 1)
        pi = rng.randint(0, k - 1)  # step index
        
        old_step = cur_steps[ri][pi]
        
        # Propose new step
        # Sample from a distribution around old_step
        if rng.random() < 0.7:
            # Local move: small change
            delta = rng.randint(-5, 5)
            new_step = old_step + delta
        else:
            # Larger move
            new_step = rng.randint(1, max(2, old_step * 2))
        
        if new_step < 1 or new_step == old_step:
            continue
        
        # Compute new row values
        new_vals = list(cur_vals[ri])
        # Recompute from step pi onward
        # new_vals[0] = 0
        # new_vals[i] for i >= pi+1 changes
        # Actually, changing step pi affects new_vals[pi+1] and all subsequent
        # Let me recompute the whole row (k is small)
        nv = [0]
        s = 0
        for j in range(k):
            if j == pi:
                s += new_step
            else:
                s += cur_steps[ri][j]
            nv.append(s)
        
        # Check strictly increasing (all steps >= 1 ensures this)
        # Get new diffs
        new_diffs = [nv[j] - nv[i] for i in range(k+1) for j in range(i+1, k+1)]
        
        # Check for conflicts:
        # 1. Internal: all new_diffs must be distinct
        if len(set(new_diffs)) != len(new_diffs):
            continue
        
        # 2. External: new_diffs must not intersect with used diffs from other rows
        # Remove old row's diffs from used, check, then restore if rejected
        old_diffs = _row_diffs_from_values(cur_vals[ri])
        
        # Check conflict with other rows' diffs
        conflict = False
        for d in new_diffs:
            if d in used and d not in old_diffs:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Compute new scope
        new_scope = max(nv) if nv[0] == 0 else max(max(nv), max(max(r) for r in cur_vals))
        # Actually scope is global max
        new_scope = max(max(nv), max(max(r) for r in cur_vals if r is not cur_vals[ri]))
        # Simpler: 
        new_scope = 0
        for idx, rv in enumerate(cur_vals):
            if idx == ri:
                m = max(nv)
            else:
                m = max(rv)
            if m > new_scope:
                new_scope = m
        
        # Accept/reject
        if new_scope < cur_scope:
            accept = True
        else:
            delta_cost = new_scope - cur_scope
            if delta_cost == 0:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta_cost / max(T, 1e-9))
        
        if accept:
            # Commit
            # Update used set
            for d in old_diffs:
                used.discard(d)
            for d in new_diffs:
                used.add(d)
            
            cur_steps[ri][pi] = new_step
            cur_vals[ri] = nv
            cur_scope = new_scope
            
            if cur_scope < best_scope:
                best_scope = cur_scope
                best_steps = [list(s) for s in cur_steps]
        
        n_iter += 1
        if n_iter % 100000 == 0:
            pass  # Could log
    
    # Convert best_steps back to rows
    best_rows = []
    for s in best_steps:
        vals = [0]
        cum = 0
        for v in s:
            cum += v
            vals.append(cum)
        best_rows.append(vals)
    
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 145.0  # Safety margin: evaluator kills at 150s
    
    rng = random.Random(42)
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
    
    # Generate initial candidates
    candidates = []
    for seed in range(5):
        r = random.Random(seed * 137 + 7)
        rows = _greedy_build(n_rows, k, r, tight=True)
        if _valid(rows, k):
            candidates.append(rows)
    
    if not candidates:
        # Fallback
        r = random.Random(0)
        candidates.append(_greedy_build(n_rows, k, r, tight=True))
    
    candidates.sort(key=_scope)
    
    # Run SA from the best candidate
    sa_deadline = time.time() + min(100.0, _DEADLINE - time.time() - 10.0)
    if sa_deadline > time.time() + 5:
        s, rows = _sa_optimize(n_rows, k, candidates[0], sa_deadline, rng)
        consider(rows)
    
    # Run SA from second candidate if time permits
    if len(candidates) > 1 and time.time() < _DEADLINE - 15:
        sa2_deadline = time.time() + min(30.0, _DEADLINE - time.time() - 5.0)
        if sa2_deadline > time.time() + 3:
            s2, rows2 = _sa_optimize(n_rows, k, candidates[1], sa2_deadline, rng)
            consider(rows2)
    
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
