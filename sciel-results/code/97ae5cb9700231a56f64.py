import time
import random
import numpy as np

_DEADLINE = 120.0  # wall-clock kill is ~150s; keep a safety margin


def _all_diffs(rows):
    out = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                out.add(r[j] - ri)
    return out


def _scope(rows):
    m = 0
    for r in rows:
        if r:
            m = max(m, r[-1])
    return m


def _greedy_one_row(k, start, used, rng):
    row = [0]
    x = start
    while len(row) < k + 1:
        x += 1
        new = [x - a for a in row]
        if not any(d in used for d in new):
            row.append(x)
            for d in new:
                used.add(d)
    return row


def _greedy_rows(n_rows, k, start=0, rng=None, max_scope=None):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_one_row(k, start, used, rng)
        if max_scope is not None and row[-1] > max_scope:
            return None
        rows.append(row)
        start = row[-1]
    return rows


def _local_search(rows, k, deadline, rng):
    rows = [list(r) for r in rows]
    n = len(rows)
    diffs = _all_diffs(rows)
    scope = _scope(rows)
    best = scope
    while time.time() < deadline:
        r = rng.randrange(n)
        row = rows[r]
        m = len(row)
        # remove an interior mark (keep 0 and the last mark)
        if m <= 2:
            continue
        idx = rng.randrange(1, m - 1)
        old = row[idx]
        old_d = [old - row[i] for i in range(idx)] + [row[j] - old for j in range(idx + 1, m)]
        for d in old_d:
            diffs.discard(d)
        del row[idx]
        # try to reinsert a mark that keeps scope <= best
        ok = False
        limit = best
        # candidate positions: just before the last element
        # search a window of candidate values
        for _ in range(200):
            # pick a random position in (0, len(row)-1)
            pos = rng.randrange(1, len(row))
            lo = row[pos - 1] + 1
            hi = row[pos] - 1
            if hi > limit:
                hi = limit
            if lo > hi:
                continue
            val = rng.randrange(lo, hi + 1)
            new_d = [val - row[i] for i in range(pos)] + [row[j] - val for j in range(pos, len(row))]
            if any(d in diffs for d in new_d):
                continue
            # commit
            row.insert(pos, val)
            for d in new_d:
                diffs.add(d)
            ok = True
            break
        if ok:
            new_scope = _scope(rows)
            if new_scope < best:
                best = new_scope
        else:
            # restore
            row.insert(idx, old)
            for d in old_d:
                diffs.add(d)
    return rows, best


def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + _DEADLINE - 10.0  # safety margin
    rng = random.Random(12345)

    # 1) quick deterministic baseline
    base = _greedy_rows(n_rows, k, start=0, rng=rng)
    if base is None:
        base = _greedy_rows(n_rows, k, start=0)
    best_rows = [list(r) for r in base]
    best_scope = _scope(best_rows)

    # 2) randomized restarts: shuffle row order + random starts
    while time.time() < deadline - 5.0:
        perm = list(range(n_rows))
        rng.shuffle(perm)
        # random small offsets between rows
        offsets = [0]
        for _ in range(n_rows - 1):
            offsets.append(rng.randrange(0, 3))
        used = set()
        rows = []
        start = 0
        ok = True
        for i in range(n_rows):
            row = _greedy_one_row(k, start, used, rng)
            if row[-1] > best_scope:
                ok = False
                break
            rows.append(row)
            start = row[-1] + offsets[i] if i < n_rows - 1 else row[-1]
        if ok and rows and _scope(rows) < best_scope:
            best_rows = [list(r) for r in rows]
            best_scope = _scope(best_rows)

    # 3) local-search polish on the best found
    if time.time() < deadline - 2.0:
        polished, new_scope = _local_search(best_rows, k, deadline - 2.0, rng)
        if new_scope < best_scope:
            best_rows = polished
            best_scope = new_scope

    return best_rows
