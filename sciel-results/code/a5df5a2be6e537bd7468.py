"""Difference triangle set (4,7) - global greedy init + local search refinement."""
import time
import random
import math

_DEADLINE = 120.0
_SAFETY = 10.0
_TIME_LIMIT = _DEADLINE - _SAFETY


def _row_diffs(row):
    d = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ri)
    return d


def _is_valid(rows):
    used = set()
    for r in rows:
        for d in _row_diffs(r):
            if d in used:
                return False
            used.add(d)
    return True


def _score(rows):
    return max(max(r) for r in rows)


def _global_greedy(n_rows, k, rng, deadline):
    """
    Build all rows interleaved: at each step, for every incomplete row find the
    smallest next value x > last such that all diffs (x - a for a in row) are
    not in the global used set. Pick the row with the smallest such x; break
    ties randomly. This produces a much tighter solution than per-row greedy.
    """
    used = set()
    rows = [[0] for _ in range(n_rows)]

    for _ in range(n_rows * k):
        if time.time() > deadline:
            return None

        best_x = float('inf')
        best_row = -1
        best_diffs = None
        tied = []

        for i in range(n_rows):
            if len(rows[i]) == k + 1:
                continue
            last = rows[i][-1]
            x = last + 1
            while True:
                valid = True
                diffs = []
                for a in rows[i]:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                    diffs.append(d)
                if valid:
                    break
                x += 1
                if x > 5000:
                    valid = False
                    break
            if not valid:
                continue
            if x < best_x:
                best_x = x
                best_row = i
                best_diffs = diffs
                tied = [i]
            elif x == best_x:
                tied.append(i)

        if best_row == -1:
            return None

        if len(tied) > 1:
            chosen = rng.choice(tied)
            # recompute diffs for chosen row
            last = rows[chosen][-1]
            x = last + 1
            while True:
                valid = True
                diffs = []
                for a in rows[chosen]:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                    diffs.append(d)
                if valid:
                    break
                x += 1
            rows[chosen].append(x)
            for d in diffs:
                used.add(d)
        else:
            rows[best_row].append(best_x)
            for d in best_diffs:
                used.add(best_diffs)

    if all(len(r) == k + 1 for r in rows):
        return rows
    return None


def _local_search(rows, k, n_rows, deadline, rng):
    """
    Local search: randomly pick a row and a position, try a new value in the
    valid range, accept if it doesn't create a collision and improves or is
    accepted via simulated annealing.
    """
    used = set()
    for r in rows:
        used.update(_row_diffs(r))

    best = _score(rows)
    best_rows = [list(r) for r in rows]

    T = 20.0
    T_min = 0.1
    cool = 0.99995

    for _ in range(2000000):
        if time.time() > deadline:
            break

        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, len(rows[ri]))
        row = rows[ri]
        lo = row[pi - 1] + 1
        hi = (row[pi + 1] - 1) if (pi + 1 < len(row)) else (max(row) + 15)
        if hi < lo:
            hi = lo
        nv = rng.randrange(lo, hi + 1)
        if nv == row[pi]:
            continue

        old = row[pi]
        # compute affected diffs
        removed = set()
        added = set()
        for i in range(len(row)):
            if i == pi:
                continue
            d_old = abs(old - row[i])
            d_new = abs(nv - row[i])
            if d_old != d_new:
                removed.add(d_old)
                added.add(d_new)

        # check collisions: added diffs must not be in used (unless also removed)
        ok = True
        for d in added:
            if d in used and d not in removed:
                ok = False
                break
        if not ok:
            continue

        # compute new score
        old_max = max(row)
        if pi == len(row) - 1 or nv > old_max:
            new_row_max = max(old_max, nv)
            others_max = max(max(r) for r in rows if r is not row)
            new_score = max(new_row_max, others_max)
        else:
            new_score = max(max(r) for r in rows)

        old_score = max(max(r) for r in rows)
        delta = new_score - old_score

        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            # accept
            row[pi] = nv
            for d in removed:
                used.discard(d)
            for d in added:
                used.add(d)

            if new_score < best:
                best = new_score
                best_rows = [list(r) for r in rows]

        T *= cool
        if T < T_min:
            T = T_min

    return best_rows, best


def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT

    rng = random.Random(42)

    # Phase 1: Global greedy with multiple randomized restarts
    best_rows = None
    best = float('inf')

    # Do several global-greedy restarts with different tie-breaking
    for attempt in range(30):
        if time.time() > deadline - 20:
            break
        r = _global_greedy(n_rows, k, rng, deadline)
        if r is not None:
            s = _score(r)
            if s < best:
                best = s
                best_rows = [list(x) for x in r]
            if best <= 112:  # counting bound, can't do better
                break

    if best_rows is None:
        # fallback: per-row greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            best_rows.append(row)
        best = _score(best_rows)

    # Phase 2: Local search refinement
    if time.time() < deadline - 5:
        ls_rows, ls_best = _local_search(
            [list(r) for r in best_rows], k, n_rows, deadline - 2, rng
        )
        if ls_best < best:
            best = ls_best
            best_rows = ls_rows

    # Final validity check
    if not _is_valid(best_rows):
        # fallback to a fresh global greedy
        r = _global_greedy(n_rows, k, rng, deadline)
        if r is not None:
            best_rows = r
        else:
            used = set()
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                best_rows.append(row)

    return best_rows
