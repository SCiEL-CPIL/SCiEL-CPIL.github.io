"""Construct difference triangle sets (n_rows, k) minimizing the largest entry.

New structural approach: formulate as a multi-objective optimization over a
global difference assignment matrix.  Use simulated annealing with a composite
energy function that penalizes both difference collisions and the square of the
maximum entry, allowing the search to navigate through invalid states to escape
local optima of strictly valid-only local search.

Key structural changes from the previous program:
  1.  Global state representation: a flat list of all row elements (excluding the
      leading 0) plus a difference-collision count per difference value.
  2.  Moves: swap two elements within a row, move an element to a different
      position in its row, or swap elements between two rows (if compatible).
  3.  Energy: E = C * n_collisions + w * (max_entry)^2, where n_collisions is
      the number of difference values that appear more than once across all rows.
  4.  Acceptance: standard Metropolis with adaptive temperature.
  5.  Final repair: once a valid state is found, use local search to reduce the
      maximum entry.
"""
import time
import random
import math

_DEADLINE = None
_TIME_LIMIT = 145.0


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


def _construct_row_partial(k, used, max_val, rng, mode):
    """Build one full row (k+1 entries, starts at 0, increasing, all <= max_val)
    with no difference in `used`. Returns the row or None on failure."""
    row = [0]
    target = k + 1
    while len(row) < target:
        last = row[-1]
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
            best = candidates[0]
            best_score = -1
            for cand in candidates[:15]:
                nd = {cand - a for a in row}
                cnt = 0
                for nxt in range(cand + 1, min(cand + 25, max_val) + 1):
                    nd2 = {nxt - a for a in row}
                    nd2.add(nxt - cand)
                    if not (nd2 & (used | nd)):
                        cnt += 1
                        if cnt >= 5:
                            break
                score = cnt * 100 - cand
                if score > best_score:
                    best_score = score
                    best = cand
            x = best
        else:  # "rand"
            pool = candidates[:min(10, len(candidates))]
            x = rng.choice(pool)
        row.append(x)
        used |= {x - a for a in row[:-1]}
    return row


def _construct_parallel(n_rows, k, rng, mode, scope_budget=None):
    """Build all rows in parallel, keeping scopes balanced."""
    rows = [[0] for _ in range(n_rows)]
    used = set()
    target = k + 1
    while any(len(r) < target for r in rows):
        order = sorted(range(n_rows), key=lambda i: (len(rows[i]), rows[i][-1]))
        for i in order:
            if len(rows[i]) >= target:
                continue
            if scope_budget is not None:
                max_val = scope_budget
                row = _construct_row_partial(k, used, max_val, rng, mode)
                if row is None:
                    return None
            else:
                last = rows[i][-1]
                chosen = None
                for cand in range(last + 1, last + 500):
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


def _energy_from_rows(rows, n_rows, k):
    """Compute energy: C * n_collisions + w * max_entry^2."""
    from collections import Counter
    diff_counter = Counter()
    max_entry = 0
    for row in rows:
        if not row:
            continue
        max_entry = max(max_entry, row[-1])
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diff_counter[d] += 1
    n_collisions = sum(1 for c in diff_counter.values() if c > 1)
    C = 1000
    w = 1
    return C * n_collisions + w * max_entry * max_entry, max_entry


def _sa_multi_objective(rows, n_rows, k, rng, max_iters=5000, initial_temp=100.0, cooling=0.995):
    """Simulated annealing with composite energy to navigate invalid states."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_energy, best_scope = _energy_from_rows(rows, n_rows, k)
    temp = initial_temp

    for it in range(max_iters):
        if _time_left() < 2.0:
            break

        # Choose a move type
        move_type = rng.random()
        i = rng.randrange(n_rows)
        row = rows[i]
        if len(row) <= 2:
            continue

        new_rows = [list(r) for r in rows]
        accepted = False

        if move_type < 0.5:
            # Swap two interior elements within a row
            if len(row) >= 3:
                p1 = rng.randrange(1, len(row) - 1)
                p2 = rng.randrange(1, len(row) - 1)
                if p1 != p2:
                    new_rows[i][p1], new_rows[i][p2] = row[p2], row[p1]
                    # Check if still increasing
                    r = new_rows[i]
                    if all(r[j] < r[j+1] for j in range(len(r)-1)):
                        accepted = True
        elif move_type < 0.8:
            # Move an element to a new random value
            pos = rng.randrange(1, len(row) - 1)
            cur = row[pos]
            lo = row[pos - 1] + 1
            hi = row[pos + 1] - 1
            if lo < hi:
                new_val = rng.randint(lo, hi)
                new_rows[i][pos] = new_val
                r = new_rows[i]
                if all(r[j] < r[j+1] for j in range(len(r)-1)):
                    accepted = True
        else:
            # Swap elements between two rows
            if n_rows >= 2:
                j = rng.randrange(n_rows)
                if j != i:
                    p1 = rng.randrange(1, len(row) - 1)
                    row_j = rows[j]
                    p2 = rng.randrange(1, len(row_j) - 1)
                    new_rows[i][p1], new_rows[j][p2] = row_j[p2], row[p1]
                    # Check both rows are still increasing
                    r1 = new_rows[i]
                    r2 = new_rows[j]
                    if (all(r1[idx] < r1[idx+1] for idx in range(len(r1)-1)) and
                            all(r2[idx] < r2[idx+1] for idx in range(len(r2)-1))):
                        accepted = True

        if not accepted:
            temp *= cooling
            continue

        new_energy, new_scope = _energy_from_rows(new_rows, n_rows, k)
        delta_e = new_energy - best_energy

        if new_energy < best_energy:
            rows = new_rows
            best_rows = [list(r) for r in rows]
            best_energy = new_energy
            best_scope = new_scope
        else:
            # Metropolis acceptance
            if temp > 0 and rng.random() < math.exp(-delta_e / temp):
                rows = new_rows
                # Don't update best unless it's valid
                if new_energy < best_energy:
                    best_rows = [list(r) for r in rows]
                    best_energy = new_energy
                    best_scope = new_scope
        temp *= cooling

    return best_rows


def _local_search_valid(rows, n_rows, k, rng, max_iters=800):
    """Local search on valid states to reduce the maximum entry."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)

    for _ in range(max_iters):
        if _time_left() < 2.0:
            break
        i = max(range(n_rows), key=lambda j: rows[j][-1])
        cur_scope = rows[i][-1]
        found = False

        # Try to reduce the last element
        row = rows[i]
        for new_last in range(cur_scope - 1, row[-2], -1):
            candidate_row = row[:-1] + [new_last]
            other_used = _all_diffs_except(rows, i)
            new_d = _row_diffs(candidate_row)
            if not (new_d & other_used):
                rows[i] = candidate_row
                if _scope(rows) < best_scope:
                    best_scope = _scope(rows)
                    best_rows = [list(r) for r in rows]
                found = True
                break

        if not found:
            # Try to reduce an interior element
            for pos in range(len(row) - 2, 0, -1):
                cur = row[pos]
                for new_val in range(cur - 1, row[pos - 1], -1):
                    if new_val >= row[pos + 1]:
                        continue
                    other_used = _all_diffs_except(rows, i)
                    candidate_row = row[:]
                    candidate_row[pos] = new_val
                    new_d = _row_diffs(candidate_row)
                    if not (new_d & other_used):
                        rows[i] = candidate_row
                        if _scope(rows) < best_scope:
                            best_scope = _scope(rows)
                            best_rows = [list(r) for r in rows]
                        found = True
                        break
                if found:
                    break

        if not found:
            # Try swapping elements between rows
            for j in range(n_rows):
                if j == i:
                    continue
                for p1 in range(1, len(rows[i])):
                    for p2 in range(1, len(rows[j])):
                        new_i = rows[i][:]
                        new_j = rows[j][:]
                        new_i[p1], new_j[p2] = new_j[p2], new_i[p1]
                        if (all(new_i[idx] < new_i[idx+1] for idx in range(len(new_i)-1)) and
                                all(new_j[idx] < new_j[idx+1] for idx in range(len(new_j)-1))):
                            other_used = set()
                            for idx in range(n_rows):
                                if idx != i and idx != j:
                                    other_used |= _row_diffs(rows[idx])
                            new_d_i = _row_diffs(new_i)
                            new_d_j = _row_diffs(new_j)
                            if not (new_d_i & other_used) and not (new_d_j & other_used) and not (new_d_i & new_d_j):
                                rows[i] = new_i
                                rows[j] = new_j
                                if _scope(rows) < best_scope:
                                    best_scope = _scope(rows)
                                    best_rows = [list(r) for r in rows]
                                found = True
                                break
                    if found:
                        break
                if found:
                    break

        if not found:
            break

    return best_rows


def _iterative_tightening(rows, n_rows, k, rng, max_rounds=40):
    """Repeatedly rebuild each row under a tighter bound than its current scope."""
    rows = [list(r) for r in rows]
    for _ in range(max_rounds):
        if _time_left() < 2.0:
            break
        improved = False
        order = sorted(range(n_rows), key=lambda i: -rows[i][-1])
        for i in order:
            if _time_left() < 2.0:
                break
            cur_scope = rows[i][-1]
            for bound in range(cur_scope - 1, max(cur_scope - 15, rows[i][-2]), -1):
                if bound <= rows[i][-2]:
                    break
                other_used = _all_diffs_except(rows, i)
                new_row = _construct_row_partial(k, other_used, bound, rng, "flex")
                if new_row is not None:
                    rows[i] = new_row
                    improved = True
                    break
        if not improved:
            break
    return rows


def construct(n_rows, k):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    rng = random.Random(42)

    # Phase 1: Build initial valid solution using parallel greedy
    best_rows = _construct_parallel(n_rows, k, rng, "flex")
    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _construct_row_partial(k, used, 10**9, rng, "flex")
            best_rows.append(row)
            used |= _row_diffs(row)
    best_scope = _scope(best_rows)

    # Phase 2: Multi-objective simulated annealing (allows invalid states)
    if _time_left() > 20.0:
        sa_rows = _sa_multi_objective(best_rows, n_rows, k, rng, max_iters=8000, initial_temp=150.0, cooling=0.998)
        if _validate(sa_rows, n_rows, k):
            sa_scope = _scope(sa_rows)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows
        # Try to repair SA result if it's close to valid
        else:
            # Use local search to try to fix it
            repaired = _local_search_valid(sa_rows, n_rows, k, rng, max_iters=200)
            if _validate(repaired, n_rows, k):
                rep_scope = _scope(repaired)
                if rep_scope < best_scope:
                    best_scope = rep_scope
                    best_rows = repaired

    # Phase 3: Local search on valid states
    if _time_left() > 15.0:
        ls = _local_search_valid(best_rows, n_rows, k, rng, max_iters=1000)
        if _scope(ls) < best_scope:
            best_scope = _scope(ls)
            best_rows = ls

    # Phase 4: Iterative tightening
    if _time_left() > 10.0:
        tightened = _iterative_tightening(best_rows, n_rows, k, rng, max_rounds=50)
        if _scope(tightened) < best_scope:
            best_scope = _scope(tightened)
            best_rows = tightened

    # Phase 5: More restarts with different strategies
    modes = ["flex", "small", "rand"]
    for attempt in range(30):
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
        if _time_left() > 10.0:
            tightened = _iterative_tightening(rows, n_rows, k, rng, max_rounds=20)
            tc = _scope(tightened)
            if tc < best_scope:
                best_scope = tc
                best_rows = tightened
            ls = _local_search_valid(tightened, n_rows, k, rng, max_iters=300)
            if _scope(ls) < best_scope:
                best_scope = _scope(ls)
                best_rows = ls

    if not _validate(best_rows, n_rows, k):
        # safe fallback
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _construct_row_partial(k, used, 10**9, rng, "flex")
            best_rows.append(row)
            used |= _row_diffs(row)

    return best_rows
