"""Difference triangle set (7,5): bitmask + randomized restarts + iterative deepening on scope."""
import time
import random
import sys

_DEADLINE = 150.0  # hard kill; keep safety margin below


def _valid_rows(rows, n_rows, k):
    """Check all positive intra-row differences are distinct across all rows."""
    used = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(k + 1):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return len(rows) == n_rows


def _construct_greedy(n_rows, k, used, rng=None):
    """Greedy: smallest next mark whose differences are all unused."""
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _construct_backtrack_scope(n_rows, k, scope, used, rng, time_limit):
    """
    Backtracking to build n_rows rows with max entry <= scope.
    Returns rows or None. Uses bitmask-free set for clarity but efficient.
    """
    rows = []

    def bt(row_idx, row, cur_x):
        if time.time() > time_limit:
            return False
        if row_idx == n_rows:
            return True
        if len(row) == k + 1:
            rows.append(row)
            return bt(row_idx + 1, [0], 0)
        # try next marks
        start = (row[-1] + 1) if row else 1
        for x in range(start, scope + 1):
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
                if bt(row_idx, row, x):
                    return True
                row.pop()
                used -= new
        return False

    if bt(0, [0], 0):
        return rows
    return None


def _construct_randomized_greedy(n_rows, k, used, rng, scope_hint=None):
    """
    Greedy with randomization: at each step, pick from a small random subset
    of valid next marks to escape bad greedy choices.
    """
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # find valid candidates in a window
            best = None
            # try a window of candidates around the smallest valid
            window = []
            scan = x + 1
            # scan up to some limit to find candidates
            limit = scan + 40
            while scan <= limit:
                new = {scan - a for a in row}
                if not (new & used):
                    window.append(scan)
                    if len(window) >= 6:
                        break
                scan += 1
            if not window:
                # fallback: keep scanning
                scan = x + 1
                while True:
                    new = {scan - a for a in row}
                    if not (new & used):
                        window.append(scan)
                        break
                    scan += 1
            # pick: mostly smallest, sometimes random
            if rng.random() < 0.7:
                pick = window[0]
            else:
                pick = window[rng.randrange(len(window))]
            row.append(pick)
            used |= {pick - a for a in row[:-1]}
            x = pick
        rows.append(row)
    return rows


def _scope_of(rows):
    return max(max(r) for r in rows) if rows else 0


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE - 10.0  # safety margin

    rng = random.Random(12345)

    # First, get a baseline greedy solution
    used = set()
    baseline_rows = _construct_greedy(n_rows, k, used)
    best_rows = baseline_rows
    best_scope = _scope_of(baseline_rows)

    # Iterative deepening on scope: try to beat best_scope
    # Lower bound: counting bound is roughly (n_rows * k*(k+1)/2)^(1/(k)) ish,
    # but we use a simple lower bound: need at least n_rows * k*(k+1)/2 distinct diffs,
    # max diff <= scope, so scope*(scope+1)/2 >= n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    # scope*(scope+1)/2 >= total_diffs
    lb = 1
    while lb * (lb + 1) // 2 < total_diffs:
        lb += 1

    # Try randomized greedy with various seeds to improve
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        # Try randomized greedy
        used = set()
        rows = _construct_randomized_greedy(n_rows, k, used, rng)
        if _valid_rows(rows, n_rows, k):
            sc = _scope_of(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [list(r) for r in rows]

        # Occasionally try backtracking with scope = best_scope - 1
        if attempts % 3 == 0 and best_scope > lb:
            target_scope = best_scope - 1
            used = set()
            rows = _construct_backtrack_scope(n_rows, k, target_scope, used, rng, deadline - 2.0)
            if rows is not None and _valid_rows(rows, n_rows, k):
                sc = _scope_of(rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = [list(r) for r in rows]

        # If we hit the lower bound, stop
        if best_scope <= lb:
            break

    return [list(r) for r in best_rows]
