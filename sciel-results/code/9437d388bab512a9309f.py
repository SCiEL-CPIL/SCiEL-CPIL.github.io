"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random
from collections import Counter

_DEADLINE = 140.0  # safety margin under the 150s wall-clock kill
_DEADLINE_SET = False


def _deadline_set():
    global _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE_SET = True
    return _DEADLINE


def construct(n_rows, k):
    start = time.time()
    deadline = _deadline_set()
    rng = random.Random(12345)

    n = n_rows
    k1 = k + 1
    m_need = k * (k + 1) // 2  # differences per row

    def build_greedy(order, used, used_set, rng=None, jitter=False):
        """Fill rows in the given order using the shared used-difference set."""
        rows = {}
        for idx in order:
            row = [0]
            x = 0
            while len(row) < k1:
                x += 1
                ok = True
                for a in row:
                    d = x - a
                    if d in used_set:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                        used_set.add(x - a)
            rows[idx] = row
            if jitter and rng is not None and len(row) > 1:
                # small random stretch on a random gap to diversify
                if rng.random() < 0.5 and len(row) >= 3:
                    gi = rng.randrange(1, len(row) - 1)
                    # rebuild is complex; skip heavy jitter, rely on ILS
        return rows

    def rows_from_dict(rows, order):
        return [rows[i] for i in order]

    def max_scope(rows):
        return max(max(r) for r in rows)

    # ---- initial solution via fast greedy (no jitter) ----
    used = set()
    used_set = used
    order0 = list(range(n))
    rows = build_greedy(order0, used, used_set)
    best_rows = rows_from_dict(rows, order0)
    best_scope = max_scope(best_rows)

    # ---- ILS: random-accept local search ----
    def all_diffs(row):
        return Counter(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))

    def try_move(row, pos, newv, used_set):
        """Return list of differences that would be removed/added if row[pos] -> newv,
        or None if the move is infeasible (duplicates or ordering violation)."""
        if pos == 0:
            return None
        if newv <= row[pos - 1] or newv >= row[pos + 1]:
            return None
        old = row[pos]
        if newv == old:
            return None
        # differences involving pos
        removed = set()
        added = set()
        for j in range(pos):
            removed.add(old - row[j])
            added.add(newv - row[j])
        for j in range(pos + 1, len(row)):
            removed.add(row[j] - old)
            added.add(row[j] - newv)
        # duplicates among added
        if len(added) != len(set(added)):
            return None
        # added must not collide with used_set minus removed
        for d in added:
            if d in used_set and d not in removed:
                return None
        return (removed, added)

    def local_search(rows, used_set, start, deadline, rng):
        # rows: list of lists (mutable). used_set: set of all diffs.
        # We maintain per-row diff sets implicitly via recomputation on move.
        # Track current scope.
        scope = max(max(r) for r in rows)
        n_rows = len(rows)
        # per-row diff multiset (differences) for quick removal
        row_diffs = [all_diffs(r) for r in rows]
        # used_set already holds everything; keep it in sync.
        it = 0
        while time.time() < deadline:
            it += 1
            # pick a random row and a random non-endpoint mark
            ri = rng.randrange(n_rows)
            row = rows[ri]
            if len(row) < 3:
                continue
            pos = rng.randrange(1, len(row) - 1)
            old = row[pos]
            # candidate new positions in a window around old
            lo = row[pos - 1] + 1
            hi = row[pos + 1] - 1
            if hi <= lo:
                continue
            span = max(1, (hi - lo))
            # choose a new value near old (small moves) or occasionally far
            if rng.random() < 0.7:
                newv = old + rng.choice([-2, -1, 1, 2, -3, 3])
            else:
                newv = rng.randrange(lo, hi + 1)
            if newv == old or newv < lo or newv > hi:
                continue
            mv = try_move(row, pos, newv, used_set)
            if mv is None:
                continue
            removed, added = mv
            # compute new scope contribution
            old_row_max = max(row)
            new_row_max = max(row) if newv < old_row_max else newv
            # Actually max(row) unchanged unless newv > old_row_max
            if newv > old_row_max:
                new_row_max = newv
            else:
                new_row_max = old_row_max
            # global new scope
            new_scope = max(new_row_max, scope) if new_row_max > scope else scope
            # accept if improves, or random accept (SA-like) to escape
            accept = False
            if new_scope < scope:
                accept = True
            else:
                # accept non-improvement with small probability to escape local minima
                if rng.random() < 0.05:
                    accept = True
            if not accept:
                continue
            # apply move
            row[pos] = newv
            rd = row_diffs[ri]
            for d in removed:
                rd[d] -= 1
                if rd[d] == 0:
                    del rd[d]
                    used_set.discard(d)
            for d in added:
                rd[d] += 1
                used_set.add(d)
            scope = new_scope
            if scope <= k * (k + 1) // 2 * n_rows - (n_rows - 1) * 0:  # just bound
                pass
        return scope

    # ---- multi-start: try several randomized row orderings ----
    best_scope = max_scope(best_rows)
    # budget: spend time on a handful of starts + ILS on the best
    elapsed = time.time() - start
    # number of multi-starts we can afford (each greedy is fast)
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        rng.shuffle(order)
        used = set()
        rows = build_greedy(order, used, used)
        sc = max_scope(rows_from_dict(rows, order))
        if sc < best_scope:
            best_scope = sc
            best_rows = rows_from_dict(rows, order)
    # ILS on the best solution (and a couple of alternates)
    candidates = [best_rows]
    # add a couple more recent good starts
    for s in range(3):
        if time.time() > deadline - 20:
            break
        order = list(range(n))
        rng.shuffle(order)
        used = set()
        rows = build_greedy(order, used, used)
        candidates.append(rows_from_dict(rows, order))
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        used_set = set()
        for r in cand:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used_set.add(r[j] - r[i])
        sc = local_search([list(r) for r in cand], used_set, time.time(), deadline, rng)
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in cand]
    return best_rows
