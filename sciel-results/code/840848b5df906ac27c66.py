import time
import random
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def _row_diffs(row):
    return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

def _validate(rows):
    used = set()
    for row in rows:
        d = _row_diffs(row)
        if d & used:
            return False
        used |= d
    return True

def _scope(rows):
    return max(max(r) for r in rows)

def _greedy_rows(n_rows, k, seed=0):
    rng = random.Random(seed)
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _backtrack_construct(n_rows, k, deadline, max_marks=200):
    """Backtracking search for n_rows rows of k+1 increasing ints starting at 0,
    minimizing the largest entry. Returns best rows found before deadline."""
    best = [None]
    best_scope = [max_marks + 1]

    used = set()

    def dfs(row_idx, row, cur_scope):
        if time.time() > deadline:
            return
        if row_idx == n_rows:
            if cur_scope < best_scope[0]:
                best_scope[0] = cur_scope
                best[0] = [list(r) for r in row]
            return
        # Prune: even if we fill remaining rows perfectly, can't beat best
        # (this is a weak bound; we rely on ordering)
        def build_row(pos, row_so_far):
            if time.time() > deadline:
                return
            if pos == k + 1:
                # row complete
                d = _row_diffs(row_so_far)
                if d & used:
                    return
                used.update(d)
                new_scope = max(cur_scope, row_so_far[-1])
                dfs(row_idx + 1, row + [row_so_far], new_scope)
                used.difference_update(d)
                return
            # Try next mark; start from last+1, but we can bound by best_scope
            start = (row_so_far[-1] + 1) if row_so_far else 1
            # Upper bound: can't exceed best_scope[0]-1 to improve
            upper = best_scope[0] if best_scope[0] < max_marks else max_marks
            for x in range(start, upper + 1):
                new_d = {x - a for a in row_so_far}
                if new_d & used:
                    continue
                # Also check within-row uniqueness is automatic since increasing
                build_row(pos + 1, row_so_far + [x])

        build_row(1, [0])

    dfs(0, [], 0)
    if best[0] is not None:
        return best[0]
    return _greedy_rows(n_rows, k)

def _local_search(rows, k, deadline, n_iters=2000):
    """Try to reduce scope via local moves. Returns best rows."""
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    used = set()
    for r in best:
        used |= _row_diffs(r)

    rng = random.Random(42)
    for _ in range(n_iters):
        if time.time() > deadline:
            break
        # Pick a random row and try to replace one of its marks (not 0, not last)
        ri = rng.randrange(len(best))
        row = best[ri]
        if len(row) < 3:
            continue
        # Pick a position to try changing
        pos = rng.randrange(1, len(row) - 1)
        old = row[pos]
        # Try a smaller value
        lo = (row[pos-1] + 1)
        hi = (row[pos+1] - 1) if pos + 1 < len(row) else old - 1
        if hi < lo:
            continue
        # Try a few candidates near old (prefer smaller)
        candidates = list(range(max(lo, old - 8), hi + 1))
        rng.shuffle(candidates)
        improved = False
        for x in candidates:
            if x == old:
                continue
            new_row = row[:pos] + [x] + row[pos+1:]
            new_d = _row_diffs(new_row)
            # Remove old row's diffs from used
            old_d = _row_diffs(row)
            # Check no conflict
            if new_d & (used - old_d):
                continue
            new_scope = max(_scope(best[:ri] + [new_row] + best[ri+1:]), )
            if new_scope < best_scope:
                # Accept
                used -= old_d
                used |= new_d
                best[ri] = new_row
                best_scope = new_scope
                improved = True
                break
            # Also accept equal scope with different structure (to escape)
            # Skip for simplicity
        if not improved and rng.random() < 0.3:
            # Try a random perturbation: swap two marks within a row
            ri = rng.randrange(len(best))
            row = best[ri]
            if len(row) < 4:
                continue
            p, q = rng.sample(range(1, len(row)), 2)
            if p > q:
                p, q = q, p
            new_row = list(row)
            new_row[p], new_row[q] = new_row[q], new_row[p]
            if new_row != sorted(new_row):
                continue
            new_d = _row_diffs(new_row)
            old_d = _row_diffs(row)
            if new_d & (used - old_d):
                continue
            used -= old_d
            used |= new_d
            best[ri] = new_row
            # scope may change
            best_scope = _scope(best)

    return best

def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin before 150s kill

    # Start with greedy to have a valid baseline
    base = _greedy_rows(n_rows, k, seed=0)
    best = [list(r) for r in base]
    best_scope = _scope(best)

    # Backtracking search with a bounded mark limit
    # We'll try increasing bounds but keep within deadline
    max_marks = best_scope
    # Try to improve via backtracking
    if time.time() < deadline - 20.0:
        bt_deadline = min(deadline - 20.0, t0 + 100.0)
        result = _backtrack_construct(n_rows, k, bt_deadline, max_marks=best_scope)
        if result is not None and _scope(result) < best_scope:
            best = result
            best_scope = _scope(best)

    # Local search polish
    if time.time() < deadline - 5.0:
        ls_deadline = deadline - 5.0
        best = _local_search(best, k, ls_deadline, n_iters=5000)
        best_scope = _scope(best)

    # Final validation
    if not _validate(best):
        best = _greedy_rows(n_rows, k, seed=0)

    return best

# EVOLVE-BLOCK-END
