"""Difference triangle set (7,5) - combined local search + row-rebuild SA."""
import random
import time
import math

_DEADLINE = 143.0  # safety margin under 150s wall-clock


def _scope(rows):
    m = 0
    for r in rows:
        if r:
            v = r[-1]
            if v > m:
                m = v
    return m


def _validate(rows):
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                if d in used:
                    return False
                used.add(d)
    return True


def _greedy(n_rows, k, rng=None):
    """Greedy construction: smallest next mark with no new difference collisions."""
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


def _build_counts(rows):
    counts = {}
    for r in rows:
        n = len(r)
        for i in range(n):
            ai = r[i]
            for j in range(i + 1, n):
                d = r[j] - ai
                counts[d] = counts.get(d, 0) + 1
    return counts


def _row_diffs_list(row):
    out = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            out.append(row[j] - ai)
    return out


def _cost(scope, counts):
    # collisions = number of diffs with count >= 2 (each such diff contributes count-1)
    coll = 0
    for c in counts.values():
        if c > 1:
            coll += c - 1
    return scope + 500 * coll


def _sa_optimize(rows, n_rows, k, deadline, rng):
    rows = [list(r) for r in rows]
    counts = _build_counts(rows)
    scope = _scope(rows)
    cost = _cost(scope, counts)

    best_rows = [list(r) for r in rows]
    best_scope = scope
    best_cost = cost

    T = 6.0
    cooling = 0.9997
    min_T = 0.05

    # precompute nothing heavy; per-move we recompute affected diffs
    while time.time() < deadline and T > min_T:
        # choose move type
        if rng.random() < 0.55:
            # ---- single-element decrement move (cheap) ----
            r = rng.randrange(n_rows)
            j = rng.randrange(1, k + 1)
            row = rows[r]
            if j >= len(row):
                continue
            old = row[j]
            if old <= row[j - 1] + 1:
                # cannot decrease
                T *= cooling
                continue
            new_val = old - 1
            # compute old and new diffs involving j
            old_diffs = []
            for i in range(j):
                old_diffs.append(old - row[i])
            for i in range(j + 1, len(row)):
                old_diffs.append(row[i] - old)
            new_diffs = []
            for i in range(j):
                new_diffs.append(new_val - row[i])
            for i in range(j + 1, len(row)):
                new_diffs.append(row[i] - new_val)

            # temporarily apply to counts
            for d in old_diffs:
                counts[d] -= 1
                if counts[d] == 0:
                    del counts[d]
            for d in new_diffs:
                counts[d] = counts.get(d, 0) + 1

            new_scope = _scope(rows) if (row[j - 1] < new_val < old and r == 0) else scope
            # recompute scope only if this row holds the max
            if old == scope:
                new_scope = _scope(rows)
            else:
                new_scope = scope
            new_cost = _cost(new_scope, counts)

            if new_cost < cost or rng.random() < math.exp(-(new_cost - cost) / T):
                row[j] = new_val
                scope = new_scope
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_scope = scope
                    best_rows = [list(x) for x in rows]
            else:
                # revert
                for d in new_diffs:
                    counts[d] -= 1
                    if counts[d] == 0:
                        del counts[d]
                for d in old_diffs:
                    counts[d] = counts.get(d, 0) + 1
                row[j] = old
        else:
            # ---- row-rebuild move (escape mechanism) ----
            r = rng.randrange(n_rows)
            old_row = list(rows[r])
            old_diffs = _row_diffs_list(old_row)
            # remove old row's diffs from counts
            for d in old_diffs:
                counts[d] -= 1
                if counts[d] == 0:
                    del counts[d]

            # rebuild row greedily, avoiding other rows' diffs (counts>0)
            max_scope = scope + 6
            new_row = [0]
            x = 0
            ok = True
            local = set()
            for _ in range(1, k + 1):
                found = False
                t = x + 1
                while t <= max_scope:
                    newd = [t - a for a in new_row]
                    conflict = False
                    for d in newd:
                        if d in counts:  # already used by another row
                            conflict = True
                            break
                    if not conflict:
                        new_row.append(t)
                        local.update(newd)
                        x = t
                        found = True
                        break
                    t += 1
                if not found:
                    ok = False
                    break
            if not ok:
                # revert: put old row back
                for d in old_diffs:
                    counts[d] = counts.get(d, 0) + 1
                rows[r] = old_row
                T *= cooling
                continue

            # add new row's diffs
            new_diffs = _row_diffs_list(new_row)
            for d in new_diffs:
                counts[d] = counts.get(d, 0) + 1

            new_scope = _scope(rows)
            new_cost = _cost(new_scope, counts)

            if new_cost < cost or rng.random() < math.exp(-(new_cost - cost) / T):
                rows[r] = new_row
                scope = new_scope
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_scope = scope
                    best_rows = [list(x) for x in rows]
            else:
                # revert
                for d in new_diffs:
                    counts[d] -= 1
                    if counts[d] == 0:
                        del counts[d]
                for d in old_diffs:
                    counts[d] = counts.get(d, 0) + 1
                rows[r] = old_row

        T *= cooling

    return best_rows, best_scope


def _final_compress(rows, n_rows, k, deadline, rng):
    """Cheap deterministic pass: repeatedly try to decrement any element by 1 if valid."""
    rows = [list(r) for r in rows]
    counts = _build_counts(rows)
    scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    best_scope = scope

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(n_rows):
            row = rows[r]
            for j in range(k, 0, -1):
                old = row[j]
                if old <= row[j - 1] + 1:
                    continue
                new_val = old - 1
                old_diffs = []
                for i in range(j):
                    old_diffs.append(old - row[i])
                for i in range(j + 1, len(row)):
                    old_diffs.append(row[i] - old)
                new_diffs = []
                for i in range(j):
                    new_diffs.append(new_val - row[i])
                for i in range(j + 1, len(row)):
                    new_diffs.append(row[i] - new_val)
                # check new diffs don't collide with counts (other rows)
                conflict = False
                for d in new_diffs:
                    # count of d excluding this row's own old occurrence
                    c = counts.get(d, 0)
                    # subtract old occurrences of d in this row
                    own = old_diffs.count(d)
                    if c - own > 0:
                        conflict = True
                        break
                if conflict:
                    continue
                # apply
                for d in old_diffs:
                    counts[d] -= 1
                    if counts[d] == 0:
                        del counts[d]
                for d in new_diffs:
                    counts[d] = counts.get(d, 0) + 1
                row[j] = new_val
                changed = True
                if old == scope:
                    scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(x) for x in rows]
    return best_rows, best_scope


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)

    # Initial valid solution via greedy
    rows = _greedy(n_rows, k, rng)
    best = [list(r) for r in rows]
    best_scope = _scope(best)

    # Main SA with row-rebuild + decrement
    if time.time() < deadline - 8:
        improved, sc = _sa_optimize(rows, n_rows, k, deadline - 8, rng)
        if sc < best_scope and _validate(improved):
            best = improved
            best_scope = sc

    # Final cheap deterministic compression
    if time.time() < deadline - 1:
        comp, sc = _final_compress(best, n_rows, k, deadline - 1, rng)
        if sc < best_scope and _validate(comp):
            best = comp
            best_scope = sc

    # Final validation; fall back to greedy if anything went wrong
    if not _validate(best):
        best = _greedy(n_rows, k, rng)

    return best
