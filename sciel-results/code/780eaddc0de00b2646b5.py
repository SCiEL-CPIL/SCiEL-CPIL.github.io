"""Difference triangle set (4,7): minimize the largest entry (scope).

Strategy: robust greedy construction (row-order + round-robin interleaved)
combined with a fast incremental-difference local search that targets the
global maximum entry.
"""
import time
import random
import sys

_START_TIME = None
_DEADLINE = 140.0  # safety margin before the 150 s kill


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE - _elapsed()


# ---------------------------------------------------------------------------
# Fast difference helpers
# ---------------------------------------------------------------------------
def _compute_diffs_set(row):
    """Return the set of all positive differences a_j - a_i (i<j) in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ai)
    return diffs


def _is_valid(rows, k):
    """Full validity check: Golomb within each row, disjoint across rows."""
    seen = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if row[j] <= row[i]:
                    return False
        d = _compute_diffs_set(row)
        if len(d) != k * (k + 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True


# ---------------------------------------------------------------------------
# Greedy constructions
# ---------------------------------------------------------------------------
def _greedy_build(n_rows, k, row_order, rng):
    """Build rows in the given order, each row greedily picks smallest valid next value."""
    used = set()
    rows = [None] * n_rows
    for r in row_order:
        row = [0]
        used_in_row = set()
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                ok = True
                for a in row:
                    d = x - a
                    if d in used or d in used_in_row:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                        used_in_row.add(x - a)
                    break
                x += 1
        rows[r] = row
    return rows


def _greedy_round_robin(n_rows, k, rng, max_val=None):
    """Round-robin: extend rows one element at a time, choosing the row with the
    smallest last element. Spreads the difference budget across all rows."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows

    while placed < total_needed:
        if _time_left() < 3.0:
            return None
        order = list(range(n_rows))
        rng.shuffle(order)
        best_r = -1
        best_last = None
        for r in order:
            if len(rows[r]) >= k + 1:
                continue
            last = rows[r][-1]
            if best_r == -1 or last < best_last:
                best_last = last
                best_r = r
        if best_r == -1:
            return None
        row = rows[best_r]
        x = row[-1] + 1
        while True:
            ok = True
            for a in row:
                if x - a in used:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
                placed += 1
                break
            x += 1
            if max_val is not None and x > max_val:
                return None
    return rows


def _unconstrained_greedy(n_rows, k):
    """Deterministic fallback that always succeeds (may be large)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new = {x - a for a in row}
                if not (new & used) and len(new) == len(row):
                    break
                x += 1
            row.append(x)
            used |= new
        rows.append(row)
    return rows


def _scope(rows):
    return max(max(r) for r in rows)


# ---------------------------------------------------------------------------
# Fast local search (incremental difference bookkeeping)
# ---------------------------------------------------------------------------
def _local_search(rows, n_rows, k, rng, time_budget):
    """Reduce the scope by pulling the global-max entry down and randomizing.

    Uses per-row difference sets so a single-element move is checked in O(k).
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = _scope(rows)
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd

    def other_diffs(r):
        s = set()
        for o in range(n_rows):
            if o != r:
                s |= row_diffs[o]
        return s

    start = time.time()
    while time.time() - start < time_budget and _time_left() > 3.0:
        # --- Targeted: pull the global-max entry downward -----------------
        max_val = _scope(rows)
        target_rows = [r for r in range(n_rows) if rows[r][-1] >= max_val - 5]
        for r in target_rows:
            if _time_left() < 3.0:
                break
            row = rows[r]
            pos = len(row) - 1
            old = row[pos]
            others = other_diffs(r)
            for nv in range(old - 1, row[pos - 1], -1):
                nd = set()
                ok = True
                for j in range(pos):
                    d = nv - row[j]
                    if d in nd or d in others:
                        ok = False
                        break
                    nd.add(d)
                if not ok:
                    continue
                # internal: nd must not collide with this row's existing diffs
                # (excluding those involving the old last element)
                base = row_diffs[r]
                for a in row[:-1]:
                    base.discard(old - a)
                if nd & base:
                    continue
                row[pos] = nv
                row_diffs[r] = base | nd
                all_diffs = set()
                for rr in range(n_rows):
                    all_diffs |= row_diffs[rr]
                cm = _scope(rows)
                if cm < best_max:
                    best_max = cm
                    best_rows = [list(rows[rr]) for rr in range(n_rows)]
                break
            else:
                row[pos] = old

        # --- Random perturbation to escape local optima -------------------
        for _ in range(40):
            if time.time() - start >= time_budget or _time_left() < 3.0:
                break
            rr = rng.randrange(n_rows)
            row = rows[rr]
            pos = rng.randrange(1, len(row))
            old = row[pos]
            lo = row[pos - 1] + 1
            hi = row[pos + 1] - 1 if pos + 1 < len(row) else old + 10
            if hi < lo:
                continue
            nv = rng.randint(lo, hi)
            if nv == old:
                continue
            others = other_diffs(rr)
            nd = set()
            ok = True
            for j in range(len(row)):
                if j == pos:
                    continue
                if j < pos:
                    d = nv - row[j]
                else:
                    d = row[j] - nv
                if d in nd or d in others:
                    ok = False
                    break
                nd.add(d)
            if not ok:
                continue
            base = row_diffs[rr]
            for j in range(len(row)):
                if j != pos:
                    base.discard(abs(old - row[j]))
            if nd & base:
                continue
            row[pos] = nv
            row_diffs[rr] = base | nd
            all_diffs = set()
            for x in range(n_rows):
                all_diffs |= row_diffs[x]
            cm = _scope(rows)
            if cm < best_max:
                best_max = cm
                best_rows = [list(rows[x]) for x in range(n_rows)]
            elif rng.random() < 0.05:
                pass  # keep (mild diversification)
            else:
                row[pos] = old
                row_diffs[rr] = base | {abs(old - row[j]) for j in range(len(row)) if j != pos}

    return best_rows, best_max


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(42)

    # 1) Deterministic valid baseline (guaranteed fallback).
    baseline = _unconstrained_greedy(n_rows, k)
    best_rows = baseline
    best_scope = _scope(baseline)

    # 2) Many diverse greedy constructions.
    for t in range(60):
        if _time_left() < 8.0:
            break
        rows = _greedy_round_robin(n_rows, k, rng)
        if rows is None:
            continue
        s = _scope(rows)
        if s < best_scope:
            best_scope = s
            best_rows = [list(r) for r in rows]

    for t in range(60):
        if _time_left() < 8.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, order, rng)
        if rows is None:
            continue
        s = _scope(rows)
        if s < best_scope:
            best_scope = s
            best_rows = [list(r) for r in rows]

    # 3) Local search: refine the best, plus a few random restarts.
    if _time_left() > 15.0:
        rows, s = _local_search(best_rows, n_rows, k, rng, min(40.0, _time_left() - 15.0))
        if s < best_scope:
            best_scope = s
            best_rows = rows

    restarts = 6
    for _ in range(restarts):
        if _time_left() < 12.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, order, rng)
        if rows is None:
            rows = _greedy_round_robin(n_rows, k, rng)
        if rows is None:
            continue
        budget = min(15.0, _time_left() - 8.0)
        if budget > 2.0:
            rows, s = _local_search(rows, n_rows, k, rng, budget)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]

    # 4) Final polish on the best with remaining time.
    remaining = _time_left() - 5.0
    if remaining > 2.0:
        best_rows, best_scope = _local_search(best_rows, n_rows, k, rng, remaining)

    # 5) Guarantee validity.
    if not _is_valid(best_rows, k):
        best_rows = _unconstrained_greedy(n_rows, k)

    return best_rows
