import time
import random
import math
import numpy as np

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

def _build_diff_counts(rows, k, max_val):
    """Build an array where diff_counts[d] = number of times difference d appears across all rows."""
    counts = np.zeros(max_val + 1, dtype=np.int32)
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts

def _row_diffs(row):
    """Return list of all differences in a single row."""
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _dfs_rebuild_row(row, k, other_diffs_set, max_scope, deadline, rng):
    """
    Try to rebuild a row to minimize its max entry, given that other_diffs_set is the set
    of differences already used by other rows. Returns (new_row, new_max) or (None, None).
    Uses DFS with pruning.
    """
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
        # Pruning: even if we pick the smallest possible values for remaining, can we beat best_max?
        # Smallest possible next values: current_x+1, current_x+2, ...
        # So the minimal possible final value is current_x + (k + 1 - pos)
        if current_x + (k + 1 - pos) >= best_max:
            return
        
        # Try candidates from smallest upward, but limit the range for speed
        # We want to find a small max, so try small values first
        start = current_x + 1
        end = min(best_max, max_scope + 5)
        
        for t in range(start, end + 1):
            ok = True
            for a in current_row:
                d = t - a
                if d <= 0:
                    continue
                if d in other_diffs_set:
                    ok = False
                    break
            if ok:
                current_row.append(t)
                # Add new diffs to a temporary set for internal conflict checking
                new_diffs = set()
                for a in current_row[:-1]:
                    d = t - a
                    new_diffs.add(d)
                # Check internal conflicts: no diff should appear twice within this row
                # Actually, the problem says "all differences between two entries of the same row are distinct over all rows"
                # Wait, re-reading: "all differences between two entries of the same row are distinct over all rows"
                # This means: for any two distinct pairs (row, i, j) and (row', i', j'), the difference must be distinct.
                # So within a row, all C(k+1, 2) differences must be distinct from each other AND from all differences in other rows.
                # So we need to ensure no diff repeats within the new row itself.
                if len(new_diffs) != len(current_row) - 1:
                    current_row.pop()
                    continue
                
                # Temporarily add to other_diffs_set for deeper recursion
                for d in new_diffs:
                    other_diffs_set.add(d)
                
                dfs(pos + 1, current_row, t)
                
                for d in new_diffs:
                    other_diffs_set.discard(d)
                current_row.pop()
            
            if time.time() > deadline:
                return
        
    dfs(1, [0], 0)
    return best_row

def _local_search(rows, k, deadline, rng):
    """
    Local search with row-rebuild moves.
    """
    n_rows = len(rows)
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)
    
    # Build global diff set and counts
    all_diffs = set()
    for r in rows:
        for d in _row_diffs(r):
            all_diffs.add(d)
    
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    stagnant = 0
    max_stagnant = 2000
    T = 10.0
    
    while time.time() < deadline:
        # Pick a random row to rebuild
        i = rng.randint(0, n_rows - 1)
        old_row = list(rows[i])
        old_max = old_row[-1]
        
        # Remove this row's diffs from global set
        row_diffs = _row_diffs(old_row)
        other_diffs_set = set(all_diffs)
        for d in row_diffs:
            other_diffs_set.discard(d)
        
        # Try to rebuild row i to minimize its max
        new_row = _dfs_rebuild_row(old_row, k, other_diffs_set, current_scope, deadline, rng)
        
        if new_row is not None:
            new_max = new_row[-1]
            new_scope = max(new_max, max(r[-1] for j, r in enumerate(rows) if j != i))
            
            # Accept if it improves scope, or with probability based on temperature
            if new_scope < current_scope:
                # Accept
                rows[i] = new_row
                # Update global diffs
                for d in row_diffs:
                    all_diffs.discard(d)
                for d in _row_diffs(new_row):
                    all_diffs.add(d)
                current_scope = new_scope
                stagnant = 0
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
            else:
                # Metropolis acceptance
                delta = new_scope - current_scope
                if rng.random() < math.exp(-delta / max(T, 0.01)):
                    rows[i] = new_row
                    for d in row_diffs:
                        all_diffs.discard(d)
                    for d in _row_diffs(new_row):
                        all_diffs.add(d)
                    current_scope = new_scope
                else:
                    stagnant += 1
        else:
            stagnant += 1
        
        # Cool down
        T = max(T * 0.999, 0.05)
        
        # Stagnation: restart from best with perturbation
        if stagnant > max_stagnant:
            rows = [list(r) for r in best_rows]
            current_scope = best_scope
            # Rebuild global diffs
            all_diffs = set()
            for r in rows:
                for d in _row_diffs(r):
                    all_diffs.add(d)
            # Perturb: randomly change one element in one row
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            prev_val = rows[i][j-1]
            next_val = rows[i][j+1] if j < k else current_scope + 100
            lo = prev_val + 1
            hi = next_val - 1
            if lo <= hi:
                rows[i][j] = rng.randint(lo, hi)
                # Check validity
                if _valid(rows, k):
                    all_diffs = set()
                    for r in rows:
                        for d in _row_diffs(r):
                            all_diffs.add(d)
                    current_scope = _scope(rows)
                else:
                    rows[i][j] = prev_val + 1  # fallback
            stagnant = 0
            T = 5.0
    
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 145.0  # Safety margin

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
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Add a few more randomized greedy candidates
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

    # Run local search on top candidates
    for idx, cand in enumerate(candidates[:5]):
        if time.time() > _DEADLINE - 5:
            break
        remaining = _DEADLINE - time.time()
        if idx == 0:
            ls_deadline = time.time() + remaining * 0.4
        elif idx == 1:
            ls_deadline = time.time() + remaining * 0.3
        elif idx == 2:
            ls_deadline = time.time() + remaining * 0.2
        else:
            ls_deadline = _DEADLINE - 5
        s, rows = _local_search(cand, k, ls_deadline, rng)
        consider(rows)

    # Final polish: run local search on best found
    if _BEST is not None and time.time() < _DEADLINE - 3:
        remaining = _DEADLINE - time.time()
        if remaining > 3:
            s, rows = _local_search(_BEST[1], k, _DEADLINE - 1, rng)
            consider(rows)

    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
