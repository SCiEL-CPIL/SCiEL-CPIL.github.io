"""New design: bounded backtracking + local search for difference-triangle set."""
import time
import random
import itertools

_DEADLINE = 115.0
_T0 = None


def _now():
    return time.time() - _T0


def _deadline_ok():
    return _now() < _DEADLINE


def _row_diffs(row):
    return set(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))


def _valid(rows, k):
    seen = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows)


def _greedy_row(used, k, scope, rng):
    """Build one row starting at 0 within [0,scope], smallest-feasible choices."""
    row = [0]
    diffs = set()
    while len(row) < k + 1:
        last = row[-1]
        cands = list(range(last + 1, scope + 1))
        rng.shuffle(cands)
        placed = False
        for x in cands:
            new = {x - a for a in row}
            if not (new & used) and not (new & diffs):
                row.append(x)
                diffs |= new
                placed = True
                break
        if not placed:
            return None
    return row


def _backtrack_row(used, k, scope, limit):
    """Backtracking to place one row within [0,scope] using unused diffs."""
    row = [0]
    diffs = set()

    def rec(idx):
        if idx == k + 1:
            return True
        last = row[-1]
        for x in range(last + 1, scope + 1):
            new = {x - a for a in row}
            if (new & used) or (new & diffs):
                continue
            row.append(x)
            diffs |= new
            if rec(idx + 1):
                return True
            row.pop()
            diffs -= new
        return False

    return rec(0)


def _construct_rows(n_rows, k, scope, rng):
    """Build n_rows rows within [0,scope] with globally distinct positive diffs."""
    used = set()
    rows = []
    for i in range(n_rows):
        # try backtracking (complete) with a node budget, else greedy
        ok = _backtrack_row(used, k, scope, None)
        if not ok:
            r = _greedy_row(used, k, scope, rng)
            if r is None:
                return None
            rows.append(r)
            used |= _row_diffs(r)
            continue
        # _backtrack_row mutated nothing externally; rebuild row by replay:
        # simpler: redo with greedy guaranteed if backtrack succeeded? We need the row.
        # Re-run a deterministic backtrack that returns the row.
        r = _backtrack_row_ret(used, k, scope)
        if r is None:
            r = _greedy_row(used, k, scope, rng)
            if r is None:
                return None
        rows.append(r)
        used |= _row_diffs(r)
    return rows


def _backtrack_row_ret(used, k, scope):
    row = [0]
    diffs = set()

    def rec(idx):
        if idx == k + 1:
            return True
        last = row[-1]
        for x in range(last + 1, scope + 1):
            new = {x - a for a in row}
            if (new & used) or (new & diffs):
                continue
            row.append(x)
            diffs |= new
            if rec(idx + 1):
                return True
            row.pop()
            diffs -= new
        return False

    if rec(0):
        return list(row)
    return None


def _local_search(rows, k, scope, rng, time_budget):
    """Try to reduce the max entry by moving marks down while keeping validity."""
    cur_scope = _scope(rows)
    end = _now() + time_budget
    while _now() < end and cur_scope > 0:
        # pick a random row and a random non-first element, try to lower it
        ri = rng.randrange(len(rows))
        r = rows[ri]
        pos = rng.randrange(1, len(r))
        old = r[pos]
        # candidate lower values
        cands = list(range(r[pos - 1] + 1, old))
        rng.shuffle(cands)
        moved = False
        for v in cands:
            trial = r[:pos] + [v] + r[pos + 1:]
            if any(trial[j] <= trial[j - 1] for j in range(1, len(trial))):
                continue
            # check global distinctness
            ok = True
            seen = set()
            for idx, rr in enumerate(rows):
                use = trial if idx == ri else rr
                ds = _row_diffs(use)
                if ds & seen:
                    ok = False
                    break
                seen |= ds
            if ok:
                rows[ri] = trial
                cur_scope = min(cur_scope, _scope(rows))
                moved = True
                break
        if not moved and cur_scope <= 0:
            break
    return rows


def construct(n_rows, k):
    global _T0
    _T0 = time.time()
    rng = random.Random(12345)

    # counting bound: need n_rows * C(k+1,2) distinct positive diffs, all <= scope
    need = n_rows * (k + 1) * k // 2
    # smallest scope s with s(s+1)/2 >= need
    s = 0
    while s * (s + 1) // 2 < need:
        s += 1
    # start a bit above bound to give slack, but not too high
    start_scope = max(s, 1)

    best = None
    # Phase 1: try to find any valid set at increasing scopes from bound upward
    for scope in range(start_scope, start_scope + 40):
        if not _deadline_ok():
            break
        for attempt in range(6):
            if not _deadline_ok():
                break
            rows = _construct_rows(n_rows, k, scope, rng)
            if rows is not None and _valid(rows, k):
                if best is None or _scope(rows) < _scope(best):
                    best = rows
                # local search to shrink further
                rows2 = _local_search(rows, k, scope, rng, time_budget=4.0)
                if _valid(rows2, k) and (best is None or _scope(rows2) < _scope(best)):
                    best = rows2
                break  # found at this scope, stop trying higher scopes
        if best is not None:
            break

    # Phase 2: if nothing found, fall back to simple greedy (always valid)
    if best is None or not _valid(best, k):
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
        best = rows

    # final safety local search with remaining time
    rem = _DEADLINE - _now()
    if rem > 3:
        best = _local_search(best, k, _scope(best), rng, time_budget=max(0, rem - 2))

    return best
