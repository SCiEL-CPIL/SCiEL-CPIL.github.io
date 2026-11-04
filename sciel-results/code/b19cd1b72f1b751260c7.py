"""Difference triangle set (7,5): minimize the largest entry (scope)."""
import time
import random

_DEADLINE = 150.0
_MARGIN = 10.0
_rng = random.Random(12345)


def _row_diffs(row):
    return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}


def _valid(rows):
    seen = set()
    for r in rows:
        if r[0] != 0:
            return False
        if any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
            return False
        for d in _row_diffs(r):
            if d in seen:
                return False
            seen.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _greedy_row(used, k, start=1):
    row = [0]
    x = start
    while len(row) < k + 1:
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
            x += 1
        else:
            x += 1
    return row


def _greedy(rows_count, k, seed):
    rng = random.Random(seed)
    order = list(range(rows_count))
    rng.shuffle(order)
    used = set()
    rows = [None] * rows_count
    for idx in order:
        rows[idx] = _greedy_row(used, k)
    return rows


def _local_search(rows, k, deadline, seed):
    rng = random.Random(seed)
    n = len(rows)
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    best_used = set()
    for r in best:
        best_used |= _row_diffs(r)

    # Precompute all positive difference slots for a row of length k+1:
    # (i, j) with i<j, difference slot value = j-i is fixed but the actual
    # numeric difference depends on the row values.
    pairs = [(i, j) for i in range(k + 1) for j in range(i + 1, k + 1)]

    def row_ok(row, used, self_set):
        # check internal duplicates
        if len(self_set) != len(pairs):
            return False
        return not (self_set & used)

    while time.time() < deadline - _MARGIN:
        # Try to shrink the largest element of the largest-scope row.
        scopes = [max(r) for r in best]
        target = max(range(n), key=lambda i: scopes[i])
        if scopes[target] <= 1:
            break
        row = best[target]
        maxval = row[-1]
        # candidate new values for the last element (must be > row[-2])
        tried = 0
        for newval in range(maxval - 1, row[-2], -1):
            if newval <= row[-2]:
                break
            cand = row[:-1] + [newval]
            self_set = _row_diffs(cand)
            if len(self_set) != len(pairs):
                continue
            others = best_used - _row_diffs(row)
            if not (self_set & others):
                # accept
                best_used = others | self_set
                best[target] = cand
                best_scope = _scope(best)
                break
            tried += 1
        # If we couldn't shrink the max, try shrinking a random interior element.
        if time.time() < deadline - _MARGIN:
            for _ in range(60):
                pos = _rng.randint(1, k)  # not the 0
                old = row[pos]
                if pos == k:
                    lo = row[pos - 1] + 1
                    hi = old
                else:
                    lo = row[pos - 1] + 1
                    hi = row[pos + 1] - 1
                if hi < lo:
                    continue
                # try smaller values
                for newval in range(old - 1, lo - 1, -1):
                    cand = row[:pos] + [newval] + row[pos + 1:]
                    self_set = _row_diffs(cand)
                    if len(self_set) != len(pairs):
                        continue
                    others = best_used - _row_diffs(row)
                    if not (self_set & others):
                        best_used = others | self_set
                        best[target] = cand
                        row = cand
                        best_scope = _scope(best)
                        break
    return best, best_scope


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE

    # Baseline: deterministic greedy (matches seed) then a couple of random orders.
    rows = _greedy(n_rows, k, seed=0)
    best = [r[:] for r in rows]
    best_scope = _scope(best)

    t = start
    seed = 1
    while t < deadline - _MARGIN:
        cand = _greedy(n_rows, k, seed=seed)
        s = _scope(cand)
        if s < best_scope:
            best = cand
            best_scope = s
        seed += 1
        t = time.time()

    # Local search to shrink scope further.
    best, best_scope = _local_search(best, k, deadline, seed=99)

    # Final safety check.
    if not _valid(best):
        # fallback to plain greedy
        best = _greedy(n_rows, k, seed=0)

    return best
