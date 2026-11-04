import time
import random

# EVOLVE-BLOCK-START
def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s

def _used_set(rows):
    used = set()
    for r in rows:
        used |= _row_diffs(r)
    return used

def _scope(rows):
    return max(max(r) for r in rows)

def _validate(rows, k):
    if len(rows) != k + 1:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        for a, b in zip(r, r[1:]):
            if b <= a:
                return False
        d = _row_diffs(r)
        if d & used:
            return False
        used |= d
    return True

def _greedy_init(n_rows, k, rng):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            ok = True
            for a in row:
                if (x - a) in used:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows

def _random_row(used, k, rng):
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += rng.randint(1, 3)
        ok = True
        for a in row:
            if (x - a) in used:
                ok = False
                break
        if ok:
            row.append(x)
    return row

def _random_init(n_rows, k, rng):
    used = set()
    rows = []
    for _ in range(n_rows):
        r = _random_row(used, k, rng)
        used |= _row_diffs(r)
        rows.append(r)
    return rows

def _local_search(rows, k, rng, deadline):
    n = len(rows)
    used = _used_set(rows)
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    T = 3.0
    cooling = 0.999
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for _ in range(n * 2):
            if time.time() >= deadline:
                break
            ri = rng.randrange(n)
            others = [r for j, r in enumerate(rows) if j != ri]
            used_o = set()
            for r in others:
                used_o |= _row_diffs(r)
            # try to rebuild row ri
            cand = [0]
            x = 0
            fail = False
            while len(cand) < k + 1:
                # small randomization in step
                x += rng.randint(1, 2)
                ok = True
                for a in cand:
                    if (x - a) in used_o or (x - a) in {cand[j] - cand[i] for i in range(len(cand)) for j in range(i + 1, len(cand))}:
                        ok = False
                        break
                if ok:
                    cand.append(x)
                else:
                    fail = True
                    break
            if fail:
                continue
            new_scope = max(cand)
            old_scope = max(rows[ri])
            delta = new_scope - old_scope
            if delta <= 0 or rng.random() < (1.0 if T > 1 else T):
                rows[ri] = cand
                used = used_o | _row_diffs(cand)
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in rows]
                improved = True
        T *= cooling
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 110.0  # safety margin under 120s
    rng = random.Random(12345)
    # multiple strategies
    best_rows = None
    best_scope = None

    # 1) greedy init + local search
    rows = _greedy_init(n_rows, k, rng)
    rows = _local_search(rows, k, rng, deadline)
    if best_scope is None or _scope(rows) < best_scope:
        best_scope = _scope(rows)
        best_rows = [list(r) for r in rows]

    # 2) random inits + local search (shuffled ordering via random init)
    while time.time() < deadline - 20.0:
        rows = _random_init(n_rows, k, rng)
        rows = _local_search(rows, k, rng, deadline)
        s = _scope(rows)
        if best_scope is None or s < best_scope:
            best_scope = s
            best_rows = [list(r) for r in rows]

    return best_rows
# EVOLVE-BLOCK-END
