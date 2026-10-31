import time
import random

_DEADLINE = 145.0
_start = time.monotonic()

def _time_left():
    return _DEADLINE - (time.monotonic() - _start)

def _diffs(row):
    n = len(row)
    out = []
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            out.append(row[j] - ai)
    return out

def _valid(rows):
    seen = set()
    for row in rows:
        for d in _diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _scope(rows):
    m = 0
    for row in rows:
        if row:
            m = max(m, row[-1])
    return m

def _greedy(n_rows, k, rng):
    used = set()
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

def _improve(rows, n_rows, k, rng):
    rows = [list(r) for r in rows]
    scope = _scope(rows)

    def build_used():
        u = set()
        for row in rows:
            u.update(_diffs(row))
        return u

    used = build_used()
    n = len(rows)
    best = scope
    improved = True
    while improved and _time_left() > 0.5:
        improved = False
        for i in range(n):
            if _time_left() < 0.5:
                break
            row = rows[i]
            L = len(row)
            for a in range(L):
                if _time_left() < 0.5:
                    break
                for b in range(a + 1, L):
                    if _time_left() < 0.5:
                        break
                    if a == 0:
                        continue
                    old = row[b]
                    if old == 0:
                        continue
                    for c in range(b, L):
                        if c == a:
                            continue
                        new = row[c]
                        if new == old:
                            continue
                        row[b], row[c] = row[c], row[b]
                        if row[b] < row[b - 1]:
                            row[b], row[c] = row[c], row[b]
                            continue
                        rowd = _diffs(row)
                        if len(set(rowd)) != len(rowd):
                            row[b], row[c] = row[c], row[b]
                            continue
                        if any(d in used for d in rowd):
                            row[b], row[c] = row[c], row[b]
                            continue
                        newscope = row[-1]
                        if newscope < best:
                            for d in rowd:
                                if d != old:
                                    used.discard(d)
                            for d in _diffs(row):
                                used.add(d)
                            best = newscope
                            improved = True
                            break
                        row[b], row[c] = row[c], row[b]
                    if _time_left() < 0.5:
                        break
    return rows, best

def _randomize(rows, rng):
    rows = [list(r) for r in rows]
    for row in rows:
        L = len(row)
        for _ in range(2):
            a = rng.randrange(1, L)
            b = rng.randrange(1, L)
            if a == b:
                continue
            row[a], row[b] = row[b], row[a]
            row.sort()
    return rows

def construct(n_rows, k):
    global _start, _DEADLINE
    _start = time.monotonic()
    _DEADLINE = _start + 145.0
    rng = random.Random(1234567)

    rows = _greedy(n_rows, k, rng)
    rows, _ = _improve(rows, n_rows, k, rng)

    while _time_left() > 2.0:
        cand = _randomize(rows, rng)
        cand, _ = _improve(cand, n_rows, k, rng)
        if _scope(cand) < _scope(rows):
            rows = cand

    rows = [list(r) for r in rows]
    if not _valid(rows):
        rows = _greedy(n_rows, k, rng)
    return rows
