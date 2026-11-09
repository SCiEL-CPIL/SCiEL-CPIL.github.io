"""Combined program: greedy bootstrap + scope-bounded backtracking + row-swap local search,
guarded by an explicit wall-clock deadline. Entry point construct(n_rows, k) is unchanged."""
import time
import random

_START = time.time()
_DEADLINE = _START + 118.0   # evaluator kills at 150s; keep a safety margin


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _row_diffs(row):
    """All positive differences a_j - a_i (i<j) of one row as a list."""
    d = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.append(row[j] - ai)
    return d


def _scope(rows):
    m = 0
    for r in rows:
        v = r[-1]
        if v > m:
            m = v
    return m


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        prev = -1
        for v in r:
            if v <= prev:
                return False
            prev = v
        for d in _row_diffs(r):
            if d in used:
                return False
            used.add(d)
    return True


def _greedy_rows(n_rows, k, rnd):
    """Simple greedy packing (no scope cap). Always returns a valid list of n_rows rows."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        local = set()
        x = 0
        while len(row) < k + 1:
            x += 1
            bad = False
            new = []
            for a in row:
                dv = x - a
                if dv in used or dv in local:
                    bad = True
                    break
                new.append(dv)
            if not bad:
                row.append(x)
                local.update(new)
                used.update(new)
        rows.append(row)
    return rows


def _backtrack_pack(n_rows, k, M, rnd, budget):
    """Try to pack n_rows rows within scope M via randomized backtracking.
    Returns a valid list of rows or None. Uses a strict time budget."""
    t_end = time.time() + budget
    perm = list(range(n_rows))
    rows = [None] * n_rows
    used = set()
    marks = []

    def rec(ridx, row_id):
        if time.time() > t_end:
            return False
        if ridx == k + 1:
            rows[row_id] = [0] + list(marks)
            return True
        start = (marks[-1] + 1) if marks else 1
        cands = list(range(start, M + 1))
        rnd.shuffle(cands)
        for x in cands:
            if time.time() > t_end:
                return False
            prev = [0] + marks
            newl = []
            bad = False
            for a in prev:
                dv = x - a
                if dv in used:
                    bad = True
                    break
                newl.append(dv)
            if bad:
                continue
            marks.append(x)
            used.update(newl)
            if rec(ridx + 1, row_id):
                return True
            used.difference_update(newl)
            marks.pop()
        return False

    for _ in range(60):
        if time.time() > t_end:
            break
        rnd.shuffle(perm)
        rows = [None] * n_rows
        used = set()
        marks = []
        ok = False
        for ri in perm:
            marks = []
            if not rec(0, ri):
                ok = False
                break
            ok = True
        if ok and all(rows[t] is not None for t in range(n_rows)):
            return rows
    return None


def _row_swap_local(rows, n_rows, k, rnd, budget):
    """Local search: repeatedly try to replace one row with a smaller-scope alternative
    (or a random one) while keeping the packing valid. Minimizes scope."""
    t_end = time.time() + budget
    rows = [list(r) for r in rows]
    used = set()
    for r in rows:
        used.update(_row_diffs(r))
    cur_scope = _scope(rows)
    best = [list(r) for r in rows]
    best_scope = cur_scope
    no_improve = 0

    while time.time() < t_end and no_improve < 200:
        ri = rnd.randrange(n_rows)
        other_used = set(used)
        for d in _row_diffs(rows[ri]):
            other_used.discard(d)
        old = rows[ri]
        old_scope = old[-1]
        cand = None
        # 1) greedy smallest-scope alternative
        row = [0]
        local = set()
        x = 0
        while len(row) < k + 1:
            x += 1
            bad = False
            new = []
            for a in row:
                dv = x - a
                if dv in other_used or dv in local:
                    bad = True
                    break
                new.append(dv)
            if not bad:
                row.append(x)
                local.update(new)
        if row[-1] < old_scope:
            cand = row
        else:
            # 2) random alternative, accept if it improves scope
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                row.append(x)
            if _row_diffs(row) and not (set(_row_diffs(row)) & other_used) and row[-1] < old_scope:
                cand = row
        if cand is not None:
            rows[ri] = cand
            used = set(other_used)
            used.update(_row_diffs(cand))
            cur_scope = _scope(rows)
            no_improve = 0
            if cur_scope < best_scope:
                best_scope = cur_scope
                best = [list(r) for r in rows]
        else:
            no_improve += 1
    return best, best_scope


def construct(n_rows, k):
    rnd = random.Random(12345)
    # 1) Always-valid baseline (simple greedy)
    base = _greedy_rows(n_rows, k, rnd)
    best = [list(r) for r in base]
    best_scope = _scope(best)

    lb = n_rows * k * (k + 1) // 2

    # 2) Bounded backtracking to find a small-scope packing (target = lb first, then grow)
    if not _dead():
        budget = min(25.0, max(3.0, _DEADLINE - _START - 30.0))
        found = None
        for M in range(lb, best_scope + 1):
            if _dead() or time.time() > _START + budget + 5.0:
                break
            rows = _backtrack_pack(n_rows, k, M, rnd, budget)
            if rows is not None and _valid(rows, n_rows, k):
                found = rows
                break
        if found is not None:
            best = [list(r) for r in found]
            best_scope = _scope(best)

    # 3) Row-swap local search to shave scope further
    if not _dead():
        rem = _DEADLINE - _START - 4.0
        if rem > 1.0:
            best, best_scope = _row_swap_local(best, n_rows, k, rnd, rem)

    # Guarantee validity before returning
    if not _valid(best, n_rows, k):
        best = base
    return best
