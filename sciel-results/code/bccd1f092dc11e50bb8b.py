"""
Difference triangle set (7,5) — structural rewrite.

Strategy:
  1. Fast greedy initialisation (same as seed) to get a valid baseline.
  2. Backtracking search with forward-checking to find a smaller scope,
     row-by-row, trying to fit each new row within a target scope.
  3. If backtracking within target fails, fall back to the greedy result.
  4. Time-guarded: returns best valid result before the 150 s wall-clock limit.
"""

import time
import random
import sys

sys.setrecursionlimit(10000)

_DEADLINE_FRACTION = 0.92  # safety margin: stop at 92 % of 150 s


def _greedy(n_rows, k, seed=0):
    """Greedy ruler construction; returns list of rows."""
    rng = random.Random(seed)
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


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)


def _validate(rows, k):
    """Check that all positive intra-row differences are distinct across rows."""
    seen = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0:
                    return False
                if d in seen:
                    return False
                seen.add(d)
    return True


def _backtrack_row(row, x, k, used, scope, deadline, depth, max_depth):
    """
    Try to extend `row` (already has len(row) entries, last is x) to length k+1,
    using only values <= scope, without reusing any difference in `used`.
    Returns (row, used) on success or None.
    """
    if time.time() > deadline:
        return None
    if len(row) == k + 1:
        return row, used
    if depth > max_depth:
        return None

    for cand in range(x + 1, scope + 1):
        new_diffs = {cand - a for a in row}
        if new_diffs & used:
            continue
        # forward check: ensure there's enough room to place remaining entries
        remaining = (k + 1) - (len(row) + 1)
        if remaining > 0 and scope - cand < remaining:
            break  # not enough room
        new_used = used | new_diffs
        row.append(cand)
        res = _backtrack_row(row, cand, k, new_used, scope, deadline, depth + 1, max_depth)
        row.pop()
        if res is not None:
            return res
    return None


def _try_scope(n_rows, k, target_scope, used, deadline, max_depth=400):
    """
    Try to complete the remaining rows so that every entry <= target_scope
    and no difference is reused.  `used` is the set of differences already
    taken by previously placed rows.  Returns (rows, used) or None.
    """
    rows = []
    current_used = used

    for _ in range(n_rows):
        row = [0]
        res = _backtrack_row(row, 0, k, current_used, target_scope, deadline, 0, max_depth)
        if res is None:
            return None
        row, current_used = res
        rows.append(row)

    return rows, current_used


def _solve_backtracking(n_rows, k, deadline):
    """
    Search for a valid set with minimal scope using backtracking.
    Returns list of rows or None.
    """
    # Lower bound on scope: counting bound
    total_diffs = n_rows * k * (k + 1) // 2
    # Need at least total_diffs distinct positive integers, so scope >= ceil((sqrt(1+8*total_diffs)-1)/2)
    import math
    lb = math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2)

    # Upper bound from greedy
    greedy_rows = _greedy(n_rows, k, seed=0)
    greedy_scope = _scope(greedy_rows)

    # Try scopes from lower bound upward
    for scope in range(lb, greedy_scope + 1):
        if time.time() > deadline:
            break
        # Try to build all rows within this scope
        res = _try_scope(n_rows, k, scope, set(), deadline, max_depth=300)
        if res is not None:
            rows, _ = res
            if _validate(rows, k):
                return rows

    return None


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 150.0 * _DEADLINE_FRACTION

    # 1. Greedy baseline
    best_rows = _greedy(n_rows, k, seed=0)
    best_scope = _scope(best_rows)

    # 2. Try a few greedy seeds to get a better baseline
    for s in range(1, 20):
        if time.time() > deadline:
            break
        rows = _greedy(n_rows, k, seed=s)
        sc = _scope(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = rows

    # 3. Backtracking search for a smaller scope
    bt_rows = _solve_backtracking(n_rows, k, deadline)
    if bt_rows is not None:
        bt_scope = _scope(bt_rows)
        if bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope

    # 4. Final validation
    if not _validate(best_rows, k):
        # Fallback: regenerate greedy
        best_rows = _greedy(n_rows, k, seed=0)

    return best_rows
