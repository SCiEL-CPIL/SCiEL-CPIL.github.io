"""Construct difference triangle sets (n_rows, k) minimizing the largest entry.

Structural approach:
  * Parallel greedy construction that keeps row scopes balanced.
  * Iterative scope-tightening: rebuild each row under a strict upper bound
    tighter than its current scope, forcing the maximum entry down.
  * Collision-driven local repair: when a rebuild fails, locate the exact
    difference collisions and repair the row, then retry with a tighter bound.
"""
import time
import random

_DEADLINE = None
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def _validate(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        prev = -1
        for v in row:
            if v <= prev:
                return False
            prev = v
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(r[-1] for r in rows)


def _row_diffs(row):
    return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}


def _all_diffs_except(rows, idx):
    used = set()
    for i, row in enumerate(rows):
        if i == idx:
            continue
        used |= _row_diffs(row)
    return used


def _greedy_row(row, used, max_val, rng, bias):
    """Complete `row` (already has 0 and is increasing) up to k+1 elements,
    all <= max_val, without introducing a difference in `used`."""
    k = len(row)  # placeholder; actual target set by caller via max_val loop
    return row


def _construct_row_partial(k, used, max_val, rng, mode):
    """Build one full row (k+1 entries, starts at 0, increasing, all <= max_val)
    with no difference in `used`. Returns the row or None on failure."""
    row = [0]
    target = k + 1
    while len(row) < target:
        last = row[-1]
        # collect valid candidates in a window
        candidates = []
        for cand in range(last + 1, max_val + 1):
            nd = {cand - a for a in row}
            if not (nd & used):
                candidates.append(cand)
        if not candidates:
            return None
        if mode == "small":
            x = candidates[0]
        elif mode == "flex":
            # pick candidate that leaves the most room for the next step
            best = candidates[0]
            best_score = -1
            for cand in candidates[:12]:
                nd = {cand - a for a in row}
                cnt = 0
                for nxt in range(cand + 1, min(cand + 20, max_val) + 1):
                    nd2 = {nxt - a for a in row}
                    nd2.add(nxt - cand)
                    if not (nd2 & (used | nd)):
                        cnt += 1
                        if cnt >= 4:
                            break
                score = cnt * 100 - cand
                if score > best_score:
                    best_score = score
                    best = cand
            x = best
        else:  # "rand"
            pool = candidates[:min(8, len(candidates))]
            x = rng.choice(pool)
        row.append(x)
        used |= {x - a for a in row[:-1]}
    return row


def _construct_parallel(n_rows, k, rng, mode, scope_budget=None):
    """Build all rows in parallel, keeping scopes balanced. If scope_budget is
    given, every row must stay <= scope_budget; otherwise balance naturally."""
    rows = [[0] for _ in range(n_rows)]
    used = set()
    target = k + 1
    while any(len(r) < target for r in rows):
        # order rows by current last value (smallest first) to balance
        order = sorted(range(n_rows), key=lambda i: (len(rows[i]), rows[i][-1]))
        for i in order:
            if len(rows[i]) >= target:
                continue
            if scope_budget is not None:
                max_val = scope_budget
            else:
                max_val = None
            if max_val is not None:
                # need to be able to reach target within budget
                row = rows[i]
                need = target - len(row)
                if max_val - row[-1] < need:
                    return None
                row = _construct_row_partial(k, used, max_val, rng, mode)
                if row is None:
                    return None
            else:
                # natural greedy with small window
                last = rows[i][-1]
                chosen = None
                for cand in range(last + 1, last + 400):
                    nd = {cand - a for a in rows[i]}
                    if not (nd & used):
                        chosen = cand
                        break
                if chosen is None:
                    return None
                row = rows[i] + [chosen]
                used |= {chosen - a for a in rows[i]}
            rows[i] = row
            used |= _row_diffs(row)
    return rows


def _rebuild_row(row_idx, rows, k, used, max_val, rng):
    """Rebuild rows[row_idx] to stay <= max_val without colliding with `used`
    (which excludes this row's own diffs). Returns new row or None."""
    other_used = _all_diffs_except(rows, row_idx)
    new_row = _construct_row_partial(k, other_used, max_val, rng, "flex")
    return new_row


def _iterative_tightening(rows, n_rows, k, rng, max_rounds=40):
    """Repeatedly rebuild each row under a tighter bound than its current scope."""
    rows = [list(r) for r in rows]
    for _ in range(max_rounds):
        if _time_left() < 2.0:
            break
        improved = False
        # rebuild rows in order of largest scope first
        order = sorted(range(n_rows), key=lambda i: -rows[i][-1])
        for i in order:
            if _time_left() < 2.0:
                break
            cur_scope = rows[i][-1]
            # try progressively tighter bounds
            for bound in range(cur_scope - 1, max(cur_scope - 12, rows[i][-2]), -1):
                if bound <= rows[i][-2]:
                    break
                new_row = _rebuild_row(i, rows, k, None, bound, rng)
                if new_row is not None:
                    rows[i] = new_row
                    improved = True
                    break
        if not improved:
            break
    return rows


def _local_search(rows, n_rows, k, rng, max_iters=400):
    """Collision-driven repair: repeatedly try to shrink the largest-scope row
    via rebuild under a tighter bound; if that fails, nudge an interior element
    to free up space, then retry."""
    rows = [list(r) for r in rows]
    for _ in range(max_iters):
        if _time_left() < 2.0:
            break
        # target the largest-scope row
        i = max(range(n_rows), key=lambda j: rows[j][-1])
        cur_scope = rows[i][-1]
        found = False
        # try to reduce last element
        for bound in range(cur_scope - 1, max(cur_scope - 10, rows[i][-2]), -1):
            new_row = _rebuild_row(i, rows, k, None, bound, rng)
            if new_row is not None:
                rows[i] = new_row
                found = True
                break
        if found:
            continue
        # try to lower an interior element to open space
        row = rows[i]
        for pos in range(len(row) - 2, 0, -1):
            cur = row[pos]
            for new_val in range(cur - 1, row[pos - 1], -1):
                if new_val >= row[pos + 1]:
                    continue
                other_used = _all_diffs_except(rows, i)
                candidate_row = row[:]
                candidate_row[pos] = new_val
                new_d = _row_diffs(candidate_row)
                if new_d & other_used:
                    continue
                rows[i] = candidate_row
                found = True
                break
            if found:
                break
        if not found:
            break
    return rows


def construct(n_rows, k):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    rng = random.Random(42)

    # Baseline: parallel greedy (natural, balanced)
    best_rows = _construct_parallel(n_rows, k, rng, "flex")
    if best_rows is None:
        # fallback pure greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _construct_row_partial(k, used, 10**9, rng, "flex")
            best_rows.append(row)
            used |= _row_diffs(row)
    best_scope = _scope(best_rows)

    # Multi-strategy restarts
    modes = ["flex", "small", "rand"]
    for attempt in range(40):
        if _time_left() < 5.0:
            break
        mode = modes[attempt % len(modes)]
        rows = _construct_parallel(n_rows, k, rng, mode)
        if rows is None:
            continue
        cur = _scope(rows)
        if cur < best_scope:
            best_scope = cur
            best_rows = rows
        # tighten
        if _time_left() > 5.0:
            tightened = _iterative_tightening(rows, n_rows, k, rng, max_rounds=15)
            tc = _scope(tightened)
            if tc < best_scope:
                best_scope = tc
                best_rows = tightened

    # Final aggressive local search
    if _time_left() > 3.0:
        ls = _local_search(best_rows, n_rows, k, rng, max_iters=500)
        if _scope(ls) < best_scope:
            best_rows = ls
            best_scope = _scope(ls)
        tightened = _iterative_tightening(best_rows, n_rows, k, rng, max_rounds=30)
        if _scope(tightened) < best_scope:
            best_rows = tightened
            best_scope = _scope(tightened)

    if not _validate(best_rows, n_rows, k):
        # safe fallback
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _construct_row_partial(k, used, 10**9, rng, "flex")
            best_rows.append(row)
            used |= _row_diffs(row)

    return best_rows
