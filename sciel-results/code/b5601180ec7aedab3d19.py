"""Difference triangle set (7,5) – combined greedy + simulated annealing + backtracking."""
import time
import random
import sys
import math

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False

_START = time.time()
_DEADLINE = _START + 142.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    """All positive differences a_j - a_i for i<j in one row."""
    d = set()
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


def _greedy_row(used, k, max_mark, rnd):
    """Greedy place one row of length k+1 starting at 0; smallest marks that keep
    all differences unused. Returns (row, used_delta, ok)."""
    row = [0]
    local = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > max_mark:
            return None, local, False
        new = []
        bad = False
        for a in row:
            dv = x - a
            if dv in used or dv in local:
                bad = True
                break
            new.append(dv)
        if not bad:
            row.append(x)
            local.update(new)
    return row, local, True


def _greedy_fill(n_rows, k, max_mark, rnd):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, local, ok = _greedy_row(used, k, max_mark, rnd)
        if not ok:
            return None, None
        rows.append(row)
        used |= local
    return rows, used


def _interleaved_greedy(n_rows, k, max_mark, rnd=None):
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
            if not new_diffs & used:
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


def _simulated_annealing(rows, n_rows, k, time_budget, rnd):
    """Simulated annealing to reduce scope. Moves: replace one row with a new
    random/greedy row that keeps the packing valid."""
    if rows is None:
        return None, None

    best = [r[:] for r in rows]
    best_scope = _scope(best)
    current = [r[:] for r in best]
    current_scope = best_scope

    # Precompute used set
    def compute_used(rows):
        used = set()
        for r in rows:
            used |= _diffs_of_row(r)
        return used

    used = compute_used(current)

    deadline = time.time() + time_budget
    T = 50.0
    cooling = 0.995
    step = 0

    while time.time() < deadline and T > 0.1:
        step += 1
        if step % 1000 == 0 and _dead():
            break

        # Pick a random row to replace
        ri = rnd.randint(0, n_rows - 1)
        old_row = current[ri]

        # Generate a new row: try greedy with slight randomization
        # Remove old row's diffs from used
        old_diffs = _diffs_of_row(old_row)
        used_without = used - old_diffs

        # Try to build a new row with max_mark = current_scope - 1 (or best_scope - 1)
        max_mark = min(current_scope, best_scope) - 1
        if max_mark < k:
            max_mark = best_scope

        # Try multiple randomizations
        new_row = None
        for attempt in range(20):
            # Slightly randomized greedy
            row = [0]
            local = set()
            x = 0
            success = True
            while len(row) < k + 1:
                x += 1
                if x > max_mark:
                    success = False
                    break
                # Sometimes skip a value to randomize
                if rnd.random() < 0.1 and x < max_mark:
                    x += 1
                    if x > max_mark:
                        success = False
                        break
                new = []
                bad = False
                for a in row:
                    dv = x - a
                    if dv in used_without or dv in local:
                        bad = True
                        break
                    new.append(dv)
                if not bad:
                    row.append(x)
                    local.update(new)
            if success and len(row) == k + 1:
                new_row = row
                break

        if new_row is None:
            # Try a completely random row
            row = [0]
            x = 0
            success = True
            while len(row) < k + 1:
                x += 1
                if x > max_mark:
                    success = False
                    break
                new = []
                bad = False
                for a in row:
                    dv = x - a
                    if dv in used_without or dv in local:
                        bad = True
                        break
                    new.append(dv)
                if not bad:
                    row.append(x)
                    local.update(new)
            if success and len(row) == k + 1:
                new_row = row

        if new_row is None:
            T *= cooling
            continue

        # Check validity of new row
        new_diffs = _diffs_of_row(new_row)
        if new_diffs & used_without:
            T *= cooling
            continue

        # Compute new scope
        new_scope = max(max(r) for i, r in enumerate(current) if i != ri)
        new_scope = max(new_scope, max(new_row))

        # Acceptance criterion
        delta = new_scope - current_scope
        if delta <= 0 or rnd.random() < math.exp(-delta / T):
            # Accept
            used = used_without | new_diffs
            current[ri] = new_row
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best = [r[:] for r in current]
        else:
            # Reject
            pass

        T *= cooling

    return best, best_scope


def _backtrack_search(n_rows, k, target, rnd, time_budget=3.0):
    """Backtracking search for a valid packing into [0..target]."""
    rows = [None] * n_rows
    used = set()

    def fill_row(row_id, marks, used_set):
        """Recursively fill one row."""
        if time.time() > time_budget:
            return False
        if len(marks) == k + 1:
            rows[row_id] = [0] + list(marks)
            return True
        start = (marks[-1] + 1) if marks else 1
        for x in range(start, target + 1):
            prev = [0] + list(marks)
            newl = set()
            bad = False
            for a in prev:
                dv = x - a
                if dv in used_set or dv in newl:
                    bad = True
                    break
                newl.add(dv)
            if bad:
                continue
            marks.append(x)
            if fill_row(row_id, marks, used_set | newl):
                return True
            marks.pop()
        return False

    for i in range(n_rows):
        if not fill_row(i, [], used):
            return None
        used |= _diffs_of_row(rows[i])
    return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rnd = random.Random(42)

    # Helper: compute all positive differences for a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper: check if a full set of rows is valid (all diffs distinct)
    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            for d in row_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True

    # Helper: compute scope (max entry)
    def scope(rows):
        return max(max(r) for r in rows)

    # Helper: count conflicts (number of duplicate differences)
    def count_conflicts(rows):
        from collections import Counter
        all_diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.append(row[j] - row[i])
        c = Counter(all_diffs)
        return sum(v - 1 for v in c.values() if v > 1)

    # Greedy construction: build rows one by one, choosing smallest next mark
    def greedy_construct(n_rows, k, start_offset=0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Interleaved greedy: build all rows simultaneously, adding one mark at a time
    def interleaved_greedy(n_rows, k):
        used = set()
        rows = [[0] for _ in range(n_rows)]
        # We need to add k more marks to each row
        # Process in rounds; in each round, for each row, try to add the smallest valid next mark
        for round_idx in range(k):
            # For each row, find the smallest x > current_max such that new diffs don't conflict
            # To balance, process rows in order but try to keep them close
            for r_idx in range(n_rows):
                row = rows[r_idx]
                current_max = row[-1]
                x = current_max
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
        return rows

    # Local search: hill climbing with random restarts
    def local_search(initial_rows, n_rows, k, time_limit=130.0):
        best_rows = [r[:] for r in initial_rows]
        best_scope = scope(best_rows)
        best_conflicts = count_conflicts(best_rows)

        current_rows = [r[:] for r in initial_rows]
        current_scope = scope(current_rows)
        current_conflicts = count_conflicts(current_rows)

        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 20000

        while time.time() < time_limit and no_improve_count < max_no_improve:
            # Choose a random row to modify
            r_idx = rnd.randint(0, n_rows - 1)
            old_row = list(current_rows[r_idx])

            # Strategy: modify the row by shifting some tail elements
            # Pick a random position to start shifting
            pos = rnd.randint(1, len(old_row) - 1)
            # Try shifting the tail by a small random amount
            shift = rnd.randint(-3, 3)
            if shift == 0:
                shift = rnd.choice([-1, 1])

            new_row = old_row[:pos] + [x + shift for x in old_row[pos:]]

            # Ensure strictly increasing and starts at 0
            valid = True
            if new_row[0] != 0:
                valid = False
            for i in range(1, len(new_row)):
                if new_row[i] <= new_row[i-1]:
                    valid = False
                    break

            if not valid:
                no_improve_count += 1
                continue

            # Compute new conflicts and scope
            # Remove old row's diffs, add new row's diffs
            old_diffs = row_diffs(old_row)
            new_diffs = row_diffs(new_row)

            # Check if new diffs conflict with other rows' diffs
            other_diffs = set()
            for i, row in enumerate(current_rows):
                if i != r_idx:
                    other_diffs |= row_diffs(row)

            # Also check within new row for duplicates (shouldn't happen if increasing)
            if len(new_diffs) != len(list(new_diffs)):
                no_improve_count += 1
                continue

            conflict_count = len(new_diffs & other_diffs)

            # Compute scope
            new_scope_val = max(max(r) for r in current_rows) if r_idx == 0 else max(current_rows[r_idx][-1], max(max(r) for i, r in enumerate(current_rows) if i != r_idx))
            # Simpler: just compute full scope
            temp_rows = list(current_rows)
            temp_rows[r_idx] = new_row
            new_scope_val = max(max(r) for r in temp_rows)

            # Accept if fewer conflicts, or same conflicts but smaller scope
            if conflict_count < current_conflicts:
                current_rows[r_idx] = new_row
                current_conflicts = conflict_count
                current_scope = new_scope_val
                no_improve_count = 0

                if current_conflicts == 0 and current_scope < best_scope:
                    best_rows = [r[:] for r in current_rows]
                    best_scope = current_scope

            elif conflict_count == current_conflicts:
                if new_scope_val < current_scope:
                    current_rows[r_idx] = new_row
                    current_scope = new_scope_val
                    no_improve_count = 0

                    if current_conflicts == 0 and current_scope < best_scope:
                        best_rows = [r[:] for r in current_rows]
                        best_scope = current_scope
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            # Occasionally try a full row replacement
            if rnd.random() < 0.1:
                r_idx2 = rnd.randint(0, n_rows - 1)
                # Generate a new greedy row for this position
                used_other = set()
                for i, row in enumerate(current_rows):
                    if i != r_idx2:
                        used_other |= row_diffs(row)

                new_row2 = [0]
                x = 0
                while len(new_row2) < k + 1:
                    x += 1
                    nd = {x - a for a in new_row2}
                    if not nd & used_other:
                        new_row2.append(x)
                        used_other |= nd

                if new_row2[-1] < current_rows[r_idx2][-1]:
                    old_diffs2 = row_diffs(current_rows[r_idx2])
                    new_diffs2 = row_diffs(new_row2)
                    other_diffs2 = set()
                    for i, row in enumerate(current_rows):
                        if i != r_idx2:
                            other_diffs2 |= row_diffs(row)
                    conf2 = len(new_diffs2 & other_diffs2)
                    if conf2 < current_conflicts:
                        current_rows[r_idx2] = new_row2
                        current_conflicts = conf2
                        current_scope = max(max(r) for r in current_rows)
                        no_improve_count = 0
                        if current_conflicts == 0 and current_scope < best_scope:
                            best_rows = [r[:] for r in current_rows]
                            best_scope = current_scope
                    elif conf2 == current_conflicts:
                        new_scope2 = max(max(r) for r in current_rows) if r_idx2 == 0 else max(new_row2[-1], max(max(r) for i, r in enumerate(current_rows) if i != r_idx2))
                        if new_scope2 < current_scope:
                            current_rows[r_idx2] = new_row2
                            current_scope = new_scope2
                            no_improve_count = 0
                            if current_conflicts == 0 and current_scope < best_scope:
                                best_rows = [r[:] for r in current_rows]
                                best_scope = current_scope
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1

        return best_rows, best_scope

    # Generate initial solutions using different strategies
    initial_solutions = []

    # Standard greedy
    initial_solutions.append(greedy_construct(n_rows, k))

    # Interleaved greedy
    initial_solutions.append(interleaved_greedy(n_rows, k))

    # Randomized greedy with small perturbations
    for _ in range(3):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a few candidates, pick one randomly among the smallest valid
                candidates = []
                for trial in range(20):
                    x += 1
                    nd = {x - a for a in row}
                    if not nd & used:
                        candidates.append(x)
                        if len(candidates) >= 3:
                            break
                if candidates:
                    x = rnd.choice(candidates)
                    nd = {x - a for a in row}
                    row.append(x)
                    used |= nd
                else:
                    # Fallback: keep incrementing until valid
                    while True:
                        x += 1
                        nd = {x - a for a in row}
                        if not nd & used:
                            row.append(x)
                            used |= nd
                            break
            rows.append(row)
        initial_solutions.append(rows)

    # Run local search on each initial solution
    best_overall = None
    best_scope_overall = float('inf')

    for init in initial_solutions:
        if time.time() > deadline - 10:
            break
        result, sc = local_search(init, n_rows, k, time_limit=deadline - time.time() - 5)
        if sc < best_scope_overall:
            best_scope_overall = sc
            best_overall = result

    # If we haven't found a valid solution, try a more aggressive search
    if best_overall is None or count_conflicts(best_overall) > 0:
        # Try simulated annealing style
        if time.time() < deadline - 5:
            current = [r[:] for r in initial_solutions[0]]
            current_conflicts = count_conflicts(current)
            current_scope = scope(current)

            temp = 50.0
            cooling = 0.999

            while time.time() < deadline - 2 and temp > 0.1:
                r_idx = rnd.randint(0, n_rows - 1)
                old_row = list(current[r_idx])
                pos = rnd.randint(1, len(old_row) - 1)
                shift = rnd.randint(-5, 5)
                if shift == 0:
                    shift = rnd.choice([-1, 1])

                new_row = old_row[:pos] + [x + shift for x in old_row[pos:]]

                valid = new_row[0] == 0 and all(new_row[i] > new_row[i-1] for i in range(1, len(new_row)))
                if not valid:
                    temp *= cooling
                    continue

                other_diffs = set()
                for i, row in enumerate(current):
                    if i != r_idx:
                        other_diffs |= row_diffs(row)

                new_diffs = row_diffs(new_row)
                conflict_count = len(new_diffs & other_diffs)

                temp_rows = list(current)
                temp_rows[r_idx] = new_row
                new_scope_val = max(max(r) for r in temp_rows)

                delta_conf = conflict_count - current_conflicts
                delta_scope = new_scope_val - current_scope
                delta = delta_conf * 10 + delta_scope

                if delta < 0 or rnd.random() < pow(2.718281828, -delta / temp):
                    current[r_idx] = new_row
                    current_conflicts = conflict_count
                    current_scope = new_scope_val

                    if current_conflicts == 0 and current_scope < best_scope_overall:
                        best_scope_overall = current_scope
                        best_overall = [r[:] for r in current]

                temp *= cooling

            if best_overall is None:
                best_overall = current
                best_scope_overall = scope(best_overall)

    # Final fallback: return the best we have
    if best_overall is None:
        best_overall = greedy_construct(n_rows, k)

    return best_overall
