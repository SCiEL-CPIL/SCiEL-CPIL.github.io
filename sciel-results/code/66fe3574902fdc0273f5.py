"""Difference triangle set (4,7): minimize the largest entry (scope)."""
import time
import random

_DEADLINE = 150.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _row_diffs_set(row):
    s = set()
    for i in range(len(row)):
        ri = row[i]
        for j in range(i + 1, len(row)):
            s.add(row[j] - ri)
    return s


def _all_diffs(rows):
    used = set()
    for r in rows:
        used |= _row_diffs_set(r)
    return used


def _valid(rows, k):
    if len(rows) != k:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(len(r)):
            if i and r[i] <= r[i - 1]:
                return False
        rs = _row_diffs_set(r)
        if rs & used:
            return False
        used |= rs
    return True


def _greedy(n_rows, k, rng):
    """Greedy rulers with a 'next candidate' skip so used differences are never re-tested."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            while True:
                new = {x - a for a in row}
                if not (new & used):
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
            x += 1
        rows.append(row)
    return rows


def _sa(rows, n_rows, k, rng, budget):
    """Swap-based simulated annealing; each move is one set intersection."""
    rows = [list(r) for r in rows]
    used = _all_diffs(rows)
    best = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]
    t = max(30.0, 0.5 * best)
    t_min = 0.05
    cool = (t_min / t) ** (1.0 / 4000.0)
    start = time.time()
    while budget > 0.0 and _time_left() > 2.0:
        ri = rng.randrange(n_rows)
        r = rows[ri]
        pos = rng.randrange(1, k + 1)
        old = r[pos]
        rdiffs = _row_diffs_set(r)
        used -= rdiffs
        new = old + rng.choice([-1, 1]) * rng.randint(1, max(1, best // 2))
        if new <= 0 or (pos + 1 < k + 1 and new >= r[pos + 1]) or (pos - 1 >= 1 and new <= r[pos - 1]):
            used |= rdiffs
            continue
        r[pos] = new
        rdiffs2 = _row_diffs_set(r)
        conflict = len(rdiffs2 & (used - rdiffs))
        if conflict == 0:
            used |= rdiffs2
            scope = max(max(rr) for rr in rows)
            if scope < best:
                best = scope
                best_rows = [list(rr) for rr in rows]
        else:
            delta = conflict - (len(rdiffs2 & rdiffs))
            if rng.random() < 0.0025 ** (delta / max(1.0, t)):
                used |= rdiffs2
            else:
                r[pos] = old
                used |= rdiffs
        t = max(t_min, t * cool)
        if (time.time() - start) > budget:
            break
    return best_rows, best


def construct(n_rows, k):
    rng = random.Random(20250607)
    best_rows, best = _greedy(n_rows, k, rng), None
    b = max(max(r) for r in best_rows)
    best = b
    while _time_left() > 6.0:
        budget = min(18.0, max(1.5, _time_left() - 6.0))
        rows, sc = _sa(best_rows, n_rows, k, rng, budget)
        if sc < best:
            best, best_rows = sc, rows
    if _time_left() > 2.0:
        rows, sc = _sa(best_rows, n_rows, k, rng, min(3.0, max(0.5, _time_left() - 1.0)))
        if sc < best:
            best, best_rows = sc, rows
    return best_rows
