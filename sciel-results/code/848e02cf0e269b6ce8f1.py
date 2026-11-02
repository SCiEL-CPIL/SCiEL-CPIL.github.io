"""Difference triangle set (n_rows, k) with deadline-guarded search."""
import time
import random

_DEADLINE_MARGIN = 5.0  # seconds to stay under the 150s kill


def _row_diffs(row):
    """Return the set of positive differences within a single row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows) if rows else 0


def _is_valid(rows, k):
    """Check that all rows have k+1 entries starting at 0, strictly increasing,
    and all positive differences are distinct over all rows."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = _row_diffs(row)
        if diffs & used:
            return False
        used |= diffs
    return True


def _greedy_row(used, k, start_hint=0):
    """Build one row greedily: pick smallest x > last such that no diff collision."""
    row = [0]
    x = max(start_hint, 1)
    while len(row) < k + 1:
        # Try x, x+1, ... until we find a valid one
        while True:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                break
            x += 1
        row.append(x)
        used |= {x - a for a in row[:-1]}
        x += 1
    return row


def _greedy_full(n_rows, k):
    """Original greedy: build rows in order, each as small as possible."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows


def _greedy_with_shuffles(n_rows, k, rng, max_iters=200):
    """Try building rows in different orders and with random perturbations."""
    best_rows = None
    best_scope = float('inf')
    for _ in range(max_iters):
        used = set()
        rows = []
        order = list(range(n_rows))
        rng.shuffle(order)
        # Sometimes start with a small random offset for the first row
        offsets = [0] * n_rows
        for _r in range(n_rows):
            if rng.random() < 0.3:
                offsets[_r] = rng.randint(0, 3)
        row_by_idx = {}
        for idx in order:
            row = _greedy_row(used, k, start_hint=offsets[idx])
            row_by_idx[idx] = row
        rows = [row_by_idx[i] for i in range(n_rows)]
        s = _scope(rows)
        if s < best_scope:
            best_scope = s
            best_rows = rows
    return best_rows


def _local_search(rows, k, rng, time_limit=10.0):
    """Attempt to reduce scope by replacing individual entries with smaller values."""
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Collect all used diffs
        used = set()
        for r in rows:
            used |= _row_diffs(r)
        # Try to reduce each entry (except index 0) by 1, 2, ...
        for ri in range(len(rows)):
            row = rows[ri]
            for pos in range(len(row) - 1, 0, -1):  # from end to start
                old_val = row[pos]
                # Try smaller values
                for new_val in range(old_val - 1, 0, -1):
                    # Must be > previous entry
                    if pos > 0 and new_val <= row[pos - 1]:
                        break
                    # Check diff collisions
                    new_diffs = {new_val - a for a in row[:pos]} | {a - new_val for a in row[pos+1:]}
                    # Remove old diffs involving this position
                    old_diffs = {old_val - a for a in row[:pos]} | {a - old_val for a in row[pos+1:]}
                    # Check if new_diffs collide with (used - old_diffs)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        # Valid move
                        used = (used - old_diffs) | new_diffs
                        row[pos] = new_val
                        improved = True
                        break
    return rows, _scope(rows)


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct over all rows.
    Minimize the largest entry (scope)."""
    t0 = time.time()
    deadline = t0 + 150.0 - _DEADLINE_MARGIN

    rng = random.Random(42)

    # Phase 1: Basic greedy
    rows = _greedy_full(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)

    # Phase 2: Try shuffled orderings
    remaining = deadline - time.time()
    if remaining > 5.0:
        iters = min(200, max(10, int(remaining / 0.05)))
        shuffled = _greedy_with_shuffles(n_rows, k, rng, max_iters=iters)
        if shuffled is not None:
            s = _scope(shuffled)
            if s < best_scope:
                best_scope = s
                best_rows = shuffled

    # Phase 3: Local search on best solution
    remaining = deadline - time.time()
    if remaining > 2.0:
        ls_time = min(remaining - 1.0, 15.0)
        if ls_time > 1.0:
            improved_rows, s = _local_search(best_rows, k, rng, time_limit=ls_time)
            if s < best_scope:
                best_scope = s
                best_rows = improved_rows

    # Phase 4: More shuffled attempts if time remains
    remaining = deadline - time.time()
    if remaining > 5.0:
        rng2 = random.Random(123)
        iters = min(300, max(10, int(remaining / 0.05)))
        shuffled = _greedy_with_shuffles(n_rows, k, rng2, max_iters=iters)
        if shuffled is not None:
            s = _scope(shuffled)
            if s < best_scope:
                best_scope = s
                best_rows = shuffled
                # Another local search
                remaining = deadline - time.time()
                if remaining > 2.0:
                    ls_time = min(remaining - 1.0, 10.0)
                    if ls_time > 1.0:
                        ir, s2 = _local_search(best_rows, k, rng2, time_limit=ls_time)
                        if s2 < best_scope:
                            best_scope = s2
                            best_rows = ir

    return best_rows
