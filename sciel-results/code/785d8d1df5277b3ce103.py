import time
import random
import math

_DEADLINE = 150.0  # evaluator wall-clock limit; we keep a safety margin below it.


def _scope_of(rows):
    return max(max(r) for r in rows)


def _greedy(n_rows, k):
    """Original greedy ruler construction."""
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


def _diffs_of_row(row):
    """All positive differences within a single row."""
    d = set()
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.add(row[j] - ai)
    return d


def _all_diffs(rows):
    s = set()
    for r in rows:
        s |= _diffs_of_row(r)
    return s


def _validate(rows, k):
    """Check every row has k+1 strictly increasing ints starting at 0 and all diffs unique."""
    seen = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        for i in range(1, len(r)):
            if r[i] <= r[i - 1]:
                return False
        d = _diffs_of_row(r)
        if d & seen:
            return False
        seen |= d
    return True


def _local_search(rows, k, time_limit):
    """Simulated annealing over mark moves/swaps to reduce scope (largest entry)."""
    rows = [list(r) for r in rows]
    n_rows = len(rows)
    start = time.time()
    deadline = start + time_limit
    best_rows = [list(r) for r in rows]
    best_scope = _scope_of(best_rows)
    cur_scope = best_scope

    # Precompute current diff set for fast validity checks.
    cur_diffs = _all_diffs(rows)

    T = 2.0
    cool = 0.9995

    while time.time() < deadline:
        # Pick a random row and a random non-zero mark to move.
        ri = random.randrange(n_rows)
        row = rows[ri]
        # Candidate positions: try to reduce the max; choose a random index > 0.
        idx = random.randrange(1, len(row))
        old_val = row[idx]

        # Suggested new value: bias toward smaller values to shrink scope.
        # Choose a random delta, sometimes negative.
        if old_val > 0 and random.random() < 0.6:
            # try to move it down
            new_val = max(1, old_val - random.randrange(1, max(2, old_val // 2 + 1)))
        else:
            new_val = old_val + random.randrange(-max(1, old_val // 4), max(2, old_val // 4))

        if new_val <= 0 or new_val == old_val:
            continue

        # Check strict increase locally: neighbors in this row.
        lo = row[idx - 1]
        hi = row[idx + 1] if idx + 1 < len(row) else None
        if new_val <= lo:
            continue
        if hi is not None and new_val >= hi:
            continue

        # Compute diff change for this row.
        old_row_diffs = _diffs_of_row(row)
        row[idx] = new_val
        new_row_diffs = _diffs_of_row(row)
        removed = old_row_diffs - new_row_diffs
        added = new_row_diffs - old_row_diffs

        # Conflict: any added diff already present in other rows or in this row's kept diffs.
        # Since we replaced the whole row's diff set, added diffs must not collide with
        # (cur_diffs - old_row_diffs) i.e. diffs outside this row.
        other_diffs = cur_diffs - old_row_diffs
        if added & other_diffs:
            row[idx] = old_val
            continue

        # Accept move.
        cur_diffs = (cur_diffs - old_row_diffs) | new_row_diffs
        new_scope = _scope_of(rows)
        delta = new_scope - cur_scope
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-6)):
            cur_scope = new_scope
            if cur_scope < best_scope:
                best_scope = cur_scope
                best_rows = [list(r) for r in rows]
        else:
            # reject
            row[idx] = old_val
            cur_diffs = (cur_diffs - new_row_diffs) | old_row_diffs

        T *= cool
        if T < 0.01:
            T = 0.01

    return best_rows


def construct(n_rows, k):
    t0 = time.time()
    # Safety margin: leave ~10s for overhead/return.
    budget = max(0.5, _DEADLINE - 10.0 - (time.time() - t0))

    # Start with greedy (fast, always valid).
    rows = _greedy(n_rows, k)
    best = [list(r) for r in rows]
    best_scope = _scope_of(best)

    # Refine with local search within the remaining budget.
    remaining = budget - (time.time() - t0)
    if remaining > 1.0:
        improved = _local_search(rows, k, remaining)
        s = _scope_of(improved)
        if s < best_scope:
            best = improved
            best_scope = s

    # Final validity guard.
    if not _validate(best, k):
        # Fall back to a fresh greedy to guarantee validity.
        best = _greedy(n_rows, k)

    return best
