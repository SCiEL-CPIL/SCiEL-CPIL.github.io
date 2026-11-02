"""Difference triangle set (7,5) — multi-start randomized greedy + targeted local search."""
import time

# EVOLVE-BLOCK-START
def _row_diffs(row):
    """Return the set of positive differences a_j - a_i (i<j) within a single row."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _valid_rows(rows, k):
    """Check that every row has k+1 increasing ints starting at 0 and all positive
    differences are distinct across all rows. Returns True if valid."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        for d in _row_diffs(row):
            if d in used:
                return False
            used.add(d)
    return True


def _scope(rows):
    """The largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_one_row(row_prefix, used, k, rng=None):
    """Given a partial row (list starting at 0) and the global used-diff set,
    greedily extend to k+1 entries. If rng is given, use randomized selection
    among the first few candidates to diversify."""
    row = list(row_prefix)
    while len(row) < k + 1:
        x = row[-1] + 1
        best = None
        # For randomized mode, consider a small window of candidates
        if rng is not None:
            window = 12
            for _ in range(window):
                new = {x - a for a in row}
                if not (new & used):
                    if best is None:
                        best = x
                    else:
                        # 50% chance to accept a later (larger) candidate
                        if rng.random() < 0.5:
                            best = x
                x += 1
            if best is None:
                # window exhausted without a valid candidate; fall through to linear scan
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not (new & used):
                        best = x
                        break
                    x += 1
        else:
            while True:
                new = {x - a for a in row}
                if not (new & used):
                    best = x
                    break
                x += 1
        row.append(best)
        used |= {best - a for a in row[:-1]}
    return row


def _randomized_greedy(n_rows, k, rng, randomize_rows=True):
    """Build all n_rows rows using randomized greedy. Optionally randomize the
    order in which rows are built to diversify the solution."""
    used = set()
    rows = []
    if randomize_rows:
        order = list(range(n_rows))
        rng.shuffle(order)
    else:
        order = list(range(n_rows))
    for _ in order:
        row = _greedy_one_row([0], used, k, rng)
        rows.append(row)
    return rows


def _shrink_largest_row(rows, k, rng, max_iters=200):
    """Local search: repeatedly find the row with the largest entry and try to
    rebuild it (with the other rows' differences fixed) to reduce the scope.
    Returns improved rows if a strictly smaller scope is found."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    for _ in range(max_iters):
        # Find the row with the largest entry
        max_idx = max(range(len(best_rows)), key=lambda i: best_rows[i][-1])
        target = best_rows[max_idx]
        if target[-1] < 2:
            break
        # Compute used diffs from all OTHER rows
        other_used = set()
        for i in range(len(best_rows)):
            if i == max_idx:
                continue
            other_used |= _row_diffs(best_rows[i])
        # Try to rebuild the target row greedily with a lower target bound
        # We want the last entry < target[-1]
        improved = False
        for attempt in range(30):
            # Randomize the greedy construction with a tighter window
            row = _greedy_one_row([0], other_used, k, rng)
            if row[-1] < target[-1]:
                # Check full validity (should hold since other_used is fixed)
                new_rows = [list(r) for r in best_rows]
                new_rows[max_idx] = row
                if _valid_rows(new_rows, k):
                    ns = _scope(new_rows)
                    if ns < best_scope:
                        best_rows = new_rows
                        best_scope = ns
                        improved = True
                        break
            else:
                # If even greedy can't beat it, try a slightly perturbed start
                # by forcing a smaller second element
                break
        if not improved:
            break
    return best_rows


def construct(n_rows, k):
    """Build n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences are distinct over all rows. Minimize the largest entry.
    Uses multi-start randomized greedy + targeted local search with a deadline
    guard."""
    import random

    deadline = time.time() + 140.0  # 140s safety margin (150s limit)
    rng = random.Random(12345)

    # Deterministic greedy baseline
    best_rows = None
    best_scope = float('inf')

    def _update_best(rows):
        nonlocal best_rows, best_scope
        if _valid_rows(rows, k):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]

    # First: deterministic greedy (the seed behavior)
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    _update_best(rows)

    # Multi-start randomized greedy + local search
    start = time.time()
    n_starts = 0
    while time.time() < deadline - 10.0:
        n_starts += 1
        # Vary the seed for each start
        rng_s = random.Random(n_starts * 7919 + 13)
        # Randomized greedy with shuffled row order
        rows = _randomized_greedy(n_rows, k, rng_s, randomize_rows=True)
        _update_best(rows)

        # Targeted local search: shrink the largest row
        if time.time() < deadline - 5.0:
            improved = _shrink_largest_row(rows, k, rng_s, max_iters=100)
            _update_best(improved)

        # If scope is already very tight, we can stop early
        # (counting bound for (7,5) is 105; known best is 112)
        if best_scope <= 112:
            break

    if best_rows is None:
        # Fallback: return the deterministic greedy result
        return rows

    return best_rows
# EVOLVE-BLOCK-END
