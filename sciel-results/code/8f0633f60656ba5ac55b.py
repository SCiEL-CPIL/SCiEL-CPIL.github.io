"""Stronger search for difference triangle set (7,5):
- Fast interleaved greedy initialisation
- Constraint-propagation-guided row repair with randomisation
- Multi-start local search with aggressive row replacement
- Simulated annealing with tail-shift and row-replacement moves
- All guarded by an explicit wall-clock deadline (142s safety margin)."""
import time
import random
import math
import sys

_START = time.time()
_DEADLINE = _START + 142.0


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    """All positive differences a_j - a_i for i<j in one row as a set."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(r[-1] for r in rows)


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
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


def _interleaved_greedy(n_rows, k, max_mark):
    """Build rows interleaved: at each step, extend the row with the smallest
    current maximum. This tends to keep all rows balanced and compact."""
    rows = [[0] for _ in range(n_rows)]
    used = set()

    for _ in range(n_rows * k):
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
            x += 1
        if not found:
            return None, None

    for r in rows:
        if len(r) != k + 1:
            return None, None
    return rows, used


def _sequential_greedy(n_rows, k):
    """Simple sequential greedy packing. Always returns a valid list."""
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


def _repair_row_greedy(row, used_other, k, max_mark, rnd=None, randomize_prob=0.0):
    """Try to complete a partial row greedily (with optional randomization).
    Returns completed row or None."""
    row = list(row)
    local = _diffs_of_row(row)
    x = row[-1] + 1
    while len(row) < k + 1:
        if x > max_mark:
            return None
        if rnd is not None and rnd.random() < randomize_prob and x < max_mark:
            x += 1
            continue
        bad = False
        for a in row:
            dv = x - a
            if dv in used_other or dv in local:
                bad = True
                break
        if not bad:
            row.append(x)
            local.update({x - a for a in row[:-1]})
            x += 1
        else:
            x += 1
    return row


def _replace_row(rows, ri, used, n_rows, k, max_mark, rnd, randomize_prob=0.0):
    """Try to replace row ri with a smaller-scope alternative.
    Returns (new_row, new_used) or (None, None)."""
    old_diffs = _diffs_of_row(rows[ri])
    used_other = used - old_diffs
    old_scope = rows[ri][-1]

    # Try to find a row with scope < old_scope
    for attempt in range(8):
        row = [0]
        local = set()
        x = 0
        success = True
        while len(row) < k + 1:
            x += 1
            if x >= old_scope:
                success = False
                break
            if rnd.random() < randomize_prob and x < old_scope - 1:
                x += 1
                if x >= old_scope:
                    success = False
                    break
            bad = False
            for a in row:
                dv = x - a
                if dv in used_other or dv in local:
                    bad = True
                    break
            if not bad:
                row.append(x)
                local.update({x - a for a in row[:-1]})
        if success and len(row) == k + 1:
            new_diffs = _diffs_of_row(row)
            if not (new_diffs & used_other):
                return row, used_other | new_diffs
    return None, None


def _hill_climb(rows, n_rows, k, time_budget, rnd):
    """Aggressive hill-climbing: repeatedly try to replace one row with a
    smaller-scope alternative."""
    t_end = time.time() + time_budget
    rows = [list(r) for r in rows]
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    cur_scope = _scope(rows)
    best = [list(r) for r in rows]
    best_scope = cur_scope
    no_improve = 0
    max_iters = 10000

    iters = 0
    while time.time() < t_end and no_improve < 150 and iters < max_iters:
        iters += 1
        ri = rnd.randrange(n_rows)
        old_scope = rows[ri][-1]

        new_row, new_used = _replace_row(rows, ri, used, n_rows, k, old_scope, rnd, 0.1)
        if new_row is not None:
            rows[ri] = new_row
            used = new_used
            cur_scope = _scope(rows)
            no_improve = 0
            if cur_scope < best_scope:
                best_scope = cur_scope
                best = [list(r) for r in rows]
        else:
            no_improve += 1

    return best, best_scope


def _simulated_annealing(rows, n_rows, k, time_budget, rnd):
    """Simulated annealing to reduce scope. Moves: replace one row, or shift
    a row's tail entries down by 1."""
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    current = [r[:] for r in best]
    current_scope = best_scope

    used = set()
    for r in current:
        used |= _diffs_of_row(r)

    deadline = time.time() + time_budget
    T = 30.0
    cooling = 0.998
    step = 0

    while time.time() < deadline and T > 0.05:
        step += 1
        if step % 100 == 0 and _dead():
            break

        move_type = rnd.random()
        if move_type < 0.6:
            # Move type 1: Replace one row
            ri = rnd.randint(0, n_rows - 1)
            old_diffs = _diffs_of_row(current[ri])
            used_without = used - old_diffs
            max_mark = min(current_scope, best_scope)

            new_row = None
            for attempt in range(15):
                row = [0]
                local = set()
                x = 0
                success = True
                while len(row) < k + 1:
                    x += 1
                    if x > max_mark:
                        success = False
                        break
                    if rnd.random() < 0.12 and x < max_mark - 1:
                        x += 1
                        if x > max_mark:
                            success = False
                            break
                    bad = False
                    for a in row:
                        dv = x - a
                        if dv in used_without or dv in local:
                            bad = True
                            break
                    if not bad:
                        row.append(x)
                        local.update({x - a for a in row[:-1]})
                if success and len(row) == k + 1:
                    new_row = row
                    break

            if new_row is None:
                T *= cooling
                continue

            new_diffs = _diffs_of_row(new_row)
            if new_diffs & used_without:
                T *= cooling
                continue

            other_max = max(max(r) for i, r in enumerate(current) if i != ri)
            new_scope = max(other_max, max(new_row))
            delta = new_scope - current_scope
            if delta <= 0 or rnd.random() < math.exp(-delta / T):
                used = used_without | new_diffs
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]
        else:
            # Move type 2: Shift a middle/tail entry down by 1
            ri = rnd.randint(0, n_rows - 1)
            row = current[ri]
            # Pick a position to shift down
            pi = rnd.randint(1, len(row) - 1)
            old_val = row[pi]
            new_val = old_val - 1
            if new_val <= row[pi - 1]:
                T *= cooling
                continue
            if pi < len(row) - 1 and new_val >= row[pi + 1]:
                T *= cooling
                continue

            old_diffs = _diffs_of_row(row)
            new_row = row[:]
            new_row[pi] = new_val
            new_diffs = _diffs_of_row(new_row)
            used_without = used - old_diffs
            if new_diffs & used_without:
                T *= cooling
                continue

            other_max = max(max(r) for i, r in enumerate(current) if i != ri)
            new_scope = max(other_max, max(new_row))
            delta = new_scope - current_scope
            if delta <= 0 or rnd.random() < math.exp(-delta / T):
                used = used_without | new_diffs
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]

        T *= cooling

    return best, best_scope


def _multi_start_hill_climb(rows, n_rows, k, time_budget, rnd, n_starts=3):
    """Run multiple hill-climbing restarts from perturbed versions of rows."""
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    t_end = time.time() + time_budget
    per_start = time_budget / n_starts

    for s in range(n_starts):
        if time.time() >= t_end:
            break
        # Perturb: randomly replace 1-2 rows
        perturbed = [list(r) for r in rows]
        for _ in range(rnd.randint(1, 2)):
            ri = rnd.randrange(n_rows)
            old_diffs = _diffs_of_row(perturbed[ri])
            used_other = set()
            for i, r in enumerate(perturbed):
                if i != ri:
                    used_other |= _diffs_of_row(r)
            max_mark = best_scope
            for attempt in range(10):
                row = [0]
                local = set()
                x = 0
                success = True
                while len(row) < k + 1:
                    x += 1
                    if x > max_mark:
                        success = False
                        break
                    if rnd.random() < 0.15 and x < max_mark - 1:
                        x += 1
                        if x > max_mark:
                            success = False
                            break
                    bad = False
                    for a in row:
                        dv = x - a
                        if dv in used_other or dv in local:
                            bad = True
                            break
                    if not bad:
                        row.append(x)
                        local.update({x - a for a in row[:-1]})
                if success and len(row) == k + 1:
                    perturbed[ri] = row
                    break

        if not _valid(perturbed, n_rows, k):
            continue

        rem = t_end - time.time()
        if rem > 1.0:
            res, res_scope = _hill_climb(perturbed, n_rows, k, min(per_start, rem), rnd)
            if res_scope < best_scope:
                best_scope = res_scope
                best = res

    return best, best_scope


def construct(n_rows, k):
    rnd = random.Random(42)

    # 1) Interleaved greedy initialization
    lb = n_rows * k * (k + 1) // 2
    best = None
    best_scope = float('inf')

    for M in [lb + 25, lb + 20, lb + 15, lb + 10, lb + 5, lb + 2, lb]:
        if _dead():
            break
        rows, used = _interleaved_greedy(n_rows, k, M)
        if rows is not None and _valid(rows, n_rows, k):
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best = [list(r) for r in rows]

    # Fallback to sequential greedy if interleaved failed
    if best is None:
        best = _sequential_greedy(n_rows, k)
        best_scope = _scope(best)

    # 2) Hill-climbing
    if not _dead():
        rem = _DEADLINE - _START - 55.0
        if rem > 2.0:
            best, best_scope = _hill_climb(best, n_rows, k, min(15.0, rem), rnd)

    # 3) Simulated annealing
    if not _dead():
        rem = _DEADLINE - _START - 40.0
        if rem > 2.0:
            best, best_scope = _simulated_annealing(best, n_rows, k, min(30.0, rem), rnd)

    # 4) Multi-start hill-climbing
    if not _dead():
        rem = _DEADLINE - _START - 25.0
        if rem > 2.0:
            best, best_scope = _multi_start_hill_climb(best, n_rows, k, min(20.0, rem), rnd, n_starts=4)

    # 5) Final polish: one more pass of hill-climbing
    if not _dead():
        rem = _DEADLINE - _START - 5.0
        if rem > 1.0:
            best, best_scope = _hill_climb(best, n_rows, k, min(10.0, rem), rnd)

    # Guarantee validity before returning
    if not _valid(best, n_rows, k):
        best = _sequential_greedy(n_rows, k)
    return best
