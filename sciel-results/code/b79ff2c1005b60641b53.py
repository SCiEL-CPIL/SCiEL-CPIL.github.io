"""Difference triangle set (n_rows, k): minimize the largest entry (scope).

Design: fast greedy initializer + time-bounded randomized restarts + a
local-search phase using O(1) difference-count lookup + a dedicated
shrink-the-max phase. Deadline guard returns the best valid result.
"""
import time
import random
import math

_DEADLINE = 150.0  # evaluator kills at 150s; keep a safety margin.
_SAFETY = 8.0      # stop searching this many seconds before the deadline.


def _row_diffs(row):
    """Set of all positive differences within a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s


def _all_diffs(rows):
    s = set()
    for r in rows:
        s |= _row_diffs(r)
    return s


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    seen = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        for a, b in zip(r, r[1:]):
            if b <= a:
                return False
        d = _row_diffs(r)
        if d & seen:
            return False
        seen |= d
    return True


def _scope(rows):
    return max(max(r) for r in rows)


def _greedy(n_rows, k, rng, target_max=None):
    """Build rows greedily, smallest next mark whose diffs are unused.

    If target_max is given, we cap how far we search for the next mark;
    if we cannot complete a row within the cap we raise a flag (return None).
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if target_max is not None and x > target_max:
                return None
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _build_with_target(n_rows, k, rng, target_max):
    """Greedy build constrained so every row stays <= target_max."""
    return _greedy(n_rows, k, rng, target_max)


def _local_search(rows, n_rows, k, rng, diff_count, time_limit):
    """Local search: move a single mark to a new value (keeping row sorted and
    diffs unique). Accept if it lowers the scope or, at equal scope, reduces
    total spread. Returns improved rows and best scope seen."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    start = time.time()
    end = start + time_limit
    n = len(rows[0])
    # Precompute per-row diffs and global diff_count (dict diff->count)
    def recompute_counts(rs):
        dc = {}
        for r in rs:
            for i in range(n):
                ai = r[i]
                for j in range(i + 1, n):
                    d = r[j] - ai
                    dc[d] = dc.get(d, 0) + 1
        return dc

    dc = recompute_counts(rows)

    def row_diffs_set(r):
        s = set()
        for i in range(n):
            ai = r[i]
            for j in range(i + 1, n):
                s.add(r[j] - ai)
        return s

    # We will operate on a working copy.
    work = [list(r) for r in rows]

    while time.time() < end:
        improved = False
        # Randomly pick a row and a mark index to try to relocate.
        for _ in range(200):
            ri = rng.randrange(n_rows)
            # pick a non-zero mark
            mi = rng.randrange(1, n)
            r = work[ri]
            old = r[mi]
            # Candidate new values: try to reduce scope if this is the max,
            # else try small shifts.
            if old == best_scope:
                # try to lower it
                candidates = []
                for v in range(old - 1, 0, -1):
                    candidates.append(v)
                rng.shuffle(candidates)
            else:
                # small local moves
                base = old
                cands = [base + d for d in range(-3, 4) if d != 0]
                cands = [c for c in cands if c > 0]
                rng.shuffle(cands)

            for newv in candidates:
                if newv <= 0:
                    continue
                # must keep row strictly increasing
                prev = r[mi - 1]
                nxt = r[mi + 1] if mi + 1 < n else None
                if newv <= prev:
                    continue
                if nxt is not None and newv >= nxt:
                    continue
                if newv == old:
                    continue
                # Compute diff change for this row.
                old_row_diffs = row_diffs_set(r)
                r[mi] = newv
                new_row_diffs = row_diffs_set(r)
                # diffs removed
                removed = old_row_diffs - new_row_diffs
                added = new_row_diffs - old_row_diffs
                # Check global uniqueness: no added diff may already be used
                # by another row (i.e., count in dc after removing this row's
                # old diffs must be 0 for added diffs).
                ok = True
                for d in removed:
                    dc[d] -= 1
                for d in added:
                    if dc.get(d, 0) > 0:
                        ok = False
                        break
                if ok:
                    for d in added:
                        dc[d] = dc.get(d, 0) + 1
                    # compute new scope
                    new_scope = max(max(rr) for rr in work)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(rr) for rr in work]
                        improved = True
                    # keep the move (it's valid), continue searching
                else:
                    # revert counts
                    for d in removed:
                        dc[d] += 1
                    r[mi] = old
                    continue
                # move accepted; reset r
                break
            # end candidate loop
            if improved and rng.random() < 0.3:
                # occasionally re-randomize to escape local minima
                pass
        # end inner attempts
        if time.time() > end:
            break

    return best_rows, best_scope


def _shrink_max(rows, n_rows, k, rng, time_limit):
    """Dedicated phase: iteratively lower the global max entry by relocating
    the mark(s) that equal the current max, searching for a valid smaller
    placement. Returns best rows and scope."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    start = time.time()
    end = start + time_limit
    n = len(rows[0])
    work = [list(r) for r in rows]

    def recompute_counts(rs):
        dc = {}
        for r in rs:
            for i in range(n):
                ai = r[i]
                for j in range(i + 1, n):
                    d = r[j] - ai
                    dc[d] = dc.get(d, 0) + 1
        return dc

    def row_diffs_set(r):
        s = set()
        for i in range(n):
            ai = r[i]
            for j in range(i + 1, n):
                s.add(r[j] - ai)
        return s

    while time.time() < end and best_scope > 0:
        dc = recompute_counts(work)
        cur_scope = max(max(r) for r in work)
        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [list(r) for r in work]
        # find all (row, index) where value == cur_scope
        targets = []
        for ri in range(n_rows):
            r = work[ri]
            for mi in range(1, n):
                if r[mi] == cur_scope:
                    targets.append((ri, mi))
        if not targets:
            break
        rng.shuffle(targets)
        progress = False
        for (ri, mi) in targets:
            if time.time() > end:
                break
            r = work[ri]
            old = r[mi]
            prev = r[mi - 1]
            nxt = r[mi + 1] if mi + 1 < n else None
            # search downward for a valid new value
            found = False
            for newv in range(old - 1, prev, -1):
                if nxt is not None and newv >= nxt:
                    continue
                old_row_diffs = row_diffs_set(r)
                r[mi] = newv
                new_row_diffs = row_diffs_set(r)
                removed = old_row_diffs - new_row_diffs
                added = new_row_diffs - old_row_diffs
                ok = True
                for d in removed:
                    dc[d] -= 1
                for d in added:
                    if dc.get(d, 0) > 0:
                        ok = False
                        break
                if ok:
                    for d in added:
                        dc[d] = dc.get(d, 0) + 1
                    found = True
                    progress = True
                    break
                else:
                    for d in removed:
                        dc[d] += 1
                    r[mi] = old
            if found:
                pass
        if not progress:
            # try a few random perturbations to escape
            for _ in range(50):
                if time.time() > end:
                    break
                ri = rng.randrange(n_rows)
                mi = rng.randrange(1, n)
                r = work[ri]
                old = r[mi]
                prev = r[mi - 1]
                nxt = r[mi + 1] if mi + 1 < n else None
                for newv in range(max(prev + 1, 1), min(nxt - 1, old + 5) + 1 if nxt else old + 5):
                    if newv == old:
                        continue
                    old_row_diffs = row_diffs_set(r)
                    r[mi] = newv
                    new_row_diffs = row_diffs_set(r)
                    removed = old_row_diffs - new_row_diffs
                    added = new_row_diffs - old_row_diffs
                    ok = True
                    for d in removed:
                        dc[d] -= 1
                    for d in added:
                        if dc.get(d, 0) > 0:
                            ok = False
                            break
                    if ok:
                        for d in added:
                            dc[d] = dc.get(d, 0) + 1
                        break
                    else:
                        for d in removed:
                            dc[d] += 1
                        r[mi] = old
            # recompute and continue
            dc = recompute_counts(work)
    return best_rows, best_scope


def construct(n_rows, k):
    start = time.time()
    deadline = start + (_DEADLINE - _SAFETY)
    rng = random.Random(12345)

    # Phase 0: baseline greedy (deterministic)
    base = _greedy(n_rows, k, rng)
    best_rows = base
    best_scope = _scope(base)

    # Phase 1: randomized greedy restarts to find a good starting point.
    while time.time() < deadline:
        # try a few randomized greedy orders via perturbation: we rebuild
        # greedily but occasionally allow a small random skip.
        rows = _greedy(n_rows, k, rng)
        if rows is None:
            break
        sc = _scope(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = rows
        # also try target-constrained build to beat current best
        tgt = best_scope - 1
        if tgt > 0:
            rows2 = _build_with_target(n_rows, k, rng, tgt)
            if rows2 is not None:
                sc2 = _scope(rows2)
                if sc2 < best_scope:
                    best_scope = sc2
                    best_rows = rows2

    # Phase 2: local search to reduce scope.
    remaining = deadline - time.time()
    if remaining > 2.0:
        ls_rows, ls_scope = _local_search(best_rows, n_rows, k, rng, None,
                                          max(0.0, remaining * 0.5))
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows

    # Phase 3: dedicated shrink-the-max phase.
    remaining = deadline - time.time()
    if remaining > 2.0:
        sh_rows, sh_scope = _shrink_max(best_rows, n_rows, k, rng,
                                        max(0.0, remaining * 0.8))
        if sh_scope < best_scope:
            best_scope = sh_scope
            best_rows = sh_rows

    # Final validity check; fall back to baseline if anything went wrong.
    if _valid(best_rows, n_rows, k):
        return best_rows
    if _valid(base, n_rows, k):
        return base
    # As a last resort, return a trivially valid construction (may be large).
    # Build rows with widely spaced marks so diffs are unique.
    rows = []
    used = set()
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
