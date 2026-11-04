"""Difference triangle set (4,7): construct n_rows rows of k+1 increasing ints from 0,
minimizing the largest entry while keeping all intra-row positive differences distinct."""
import time
import random
import numpy as np

def _row_diffs(row):
    """Return the set of positive differences within a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs

def _all_diffs(rows):
    """Return the set of all positive differences across all rows."""
    used = set()
    for r in rows:
        used |= _row_diffs(r)
    return used

def _valid(rows):
    """Check that all intra-row positive differences are distinct across rows."""
    used = set()
    for r in rows:
        d = _row_diffs(r)
        if d & used:
            return False
        used |= d
    return True

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_row(used, k, rng):
    """Build one row of k+1 entries starting at 0, choosing smallest valid next mark.
    used is the set of already-used differences (from previous rows).
    Returns the row list."""
    row = [0]
    while len(row) < k + 1:
        x = row[-1] + 1
        while True:
            new = {x - a for a in row}
            if not (new & used):
                break
            x += 1
        row.append(x)
        used |= {x - a for a in row[:-1]}
    return row

def _pure_greedy(n_rows, k):
    """Deterministic greedy: each row takes the smallest available marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new = {x - a for a in row}
                if not (new & used):
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        rows.append(row)
    return rows

def _randomized_greedy(n_rows, k, rng, bias=0.15):
    """Randomized greedy: with probability bias, skip to a larger mark to diversify.
    Returns a valid set of rows."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            # Find the smallest valid x
            base_x = x
            while True:
                new = {x - a for a in row}
                if not (new & used):
                    break
                x += 1
            # With some probability, jump ahead to a larger valid mark
            if rng.random() < bias:
                jump = rng.randint(1, 5)
                x = base_x + jump
                while True:
                    new = {x - a for a in row}
                    if not (new & used):
                        break
                    x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        rows.append(row)
    return rows

def _local_search_shrink(rows, k, rng, max_iter=200):
    """Try to reduce the scope by re-optimizing rows. Returns improved rows or None."""
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    used = _all_diffs(best)
    
    for _ in range(max_iter):
        # Pick a random row to re-optimize
        idx = rng.randint(0, len(best) - 1)
        # Remove this row's diffs from used
        row_diffs = _row_diffs(best[idx])
        used -= row_diffs
        
        # Rebuild this row greedily with the remaining used set
        old_row = best[idx]
        new_row = [0]
        while len(new_row) < k + 1:
            x = new_row[-1] + 1
            while True:
                new = {x - a for a in new_row}
                if not (new & used):
                    break
                x += 1
            new_row.append(x)
            used |= {x - a for a in new_row[:-1]}
        
        # Check if scope improved
        new_scope = max(max(r) for r in best) if idx != len(best)-1 else max(max(best[:idx]+[new_row]+best[idx+1:]))
        # Actually recompute properly
        trial = [r if i != idx else new_row for i, r in enumerate(best)]
        trial_scope = _scope(trial)
        
        if trial_scope < best_scope:
            best = trial
            best_scope = trial_scope
            used = _all_diffs(best)
        else:
            # Revert
            used |= row_diffs  # restore (but this may be wrong if new_row added different diffs)
            # Better: just recompute used
            used = _all_diffs(best)
    
    return best

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Start with pure greedy
    rows = _pure_greedy(n_rows, k)
    best_scope = _scope(rows)
    best_rows = [r[:] for r in rows]
    
    rng = random.Random(42)
    
    # Try randomized greedy multiple times
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        bias = min(0.05 + 0.02 * attempts, 0.3)
        cand = _randomized_greedy(n_rows, k, rng, bias=bias)
        if _valid(cand):
            s = _scope(cand)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in cand]
        
        # Also try local search from current best
        if attempts % 5 == 0 and time.time() < deadline:
            cand2 = _local_search_shrink(best_rows, k, rng, max_iter=50)
            if cand2 and _valid(cand2):
                s2 = _scope(cand2)
                if s2 < best_scope:
                    best_scope = s2
                    best_rows = cand2
    
    return best_rows
