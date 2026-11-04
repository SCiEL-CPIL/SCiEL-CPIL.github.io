"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _total_scope(rows):
    """Return the maximum entry across all rows."""
    m = 0
    for r in rows:
        if r and r[-1] > m:
            m = r[-1]
    return m


def _diffs_of_row(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs_used(rows):
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    return used


def _greedy_row(used, k):
    """Greedily build one row of k+1 increasing ints starting at 0."""
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, new_diffs


def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, new_diffs = _greedy_row(used, k)
        rows.append(row)
        used |= new_diffs
    return rows


def _randomized_greedy_row(used, k, rng, pool_size=5):
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        if rng.random() < 0.6:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, new_diffs


def _randomized_greedy(n_rows, k, rng, pool_size=5):
    best_rows = None
    best_scope = float('inf')
    deadline = time.time() + 100
    for _ in range(50):
        if time.time() > deadline:
            break
        used = set()
        rows = []
        ok = True
        for _ in range(n_rows):
            row, new_diffs = _randomized_greedy_row(used, k, rng, pool_size)
            if row is None:
                ok = False
                break
            rows.append(row)
            used |= new_diffs
        if ok:
            scope = _total_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    return best_rows if best_rows is not None else _greedy_construct(n_rows, k)


def _local_search(rows, k, rng, max_iterations=5000):
    """Pure-descent local search: shrink the largest elements greedily."""
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    for _ in range(max_iterations):
        max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
        row = rows[max_idx]
        if len(row) < 2:
            break
        last = row[-1]
        found = False
        for new_last in range(last - 1, row[-2], -1):
            old_diffs = {last - a for a in row[:-1]}
            new_diffs = {new_last - a for a in row[:-1]}
            if not (new_diffs & (used - old_diffs)):
                row[-1] = new_last
                used = (used - old_diffs) | new_diffs
                found = True
                break
        if not found:
            for pos in range(len(row) - 2, 0, -1):
                old_val = row[pos]
                f2 = False
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        old_diffs.add(abs(row[i] - old_val))
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        new_diffs.add(abs(row[i] - new_val))
                    if not (new_diffs & (used - old_diffs)):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        f2 = True
                        break
                if f2:
                    break
            if not f2:
                break
    return rows


def _sa_search(rows, k, rng, deadline):
    """
    Simulated annealing with a fast incremental validity model.

    We track the multiset of all pairwise differences (a Counter). A solution
    is valid iff every count is exactly 1. Moving one element changes only the
    differences that involve it, so we update the counter incrementally and
    keep a running 'bad' count (number of differences whose count != 1).
    Acceptance uses the SA rule on the scope (max entry), so scope-increasing
    moves can be accepted to escape local optima.
    """
    import collections
    from math import exp

    rows = [r[:] for r in rows]
    n_rows = len(rows)
    counter = collections.Counter()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                counter[r[j] - r[i]] += 1
    bad = sum(1 for v in counter.values() if v != 1)
    current_scope = _total_scope(rows)
    best_scope = current_scope
    best_rows = [r[:] for r in rows]

    temp = 60.0
    cool = 0.99
    min_temp = 0.5
    step_lo, step_hi = -4, 4

    while temp > min_temp:
        if time.time() > deadline:
            break
        row_idx = rng.randrange(n_rows)
        row = rows[row_idx]
        m = len(row)
        elem = rng.randrange(1, m)
        old_val = row[elem]
        step = rng.randint(step_lo, step_hi)
        if step == 0:
            continue
        new_val = old_val + step
        if new_val <= row[elem - 1]:
            continue
        if elem < m - 1 and new_val >= row[elem + 1]:
            continue

        # Differences that involve 'elem'.
        changed = []
        for i in range(m):
            if i == elem:
                continue
            changed.append(abs(row[i] - old_val))
        row[elem] = new_val
        new_changed = []
        for i in range(m):
            if i == elem:
                continue
            new_changed.append(abs(row[i] - new_val))

        # Update the counter and the running 'bad' count.
        for d in changed:
            c = counter[d]
            if c == 1:
                bad += 1
            elif c == 2:
                bad -= 1
            counter[d] = c - 1
            if counter[d] == 0:
                del counter[d]
        for d in new_changed:
            c = counter.get(d, 0)
            if c == 1:
                bad += 1
            elif c == 0:
                bad -= 1
            counter[d] = c + 1

        if bad == 0:
            new_scope = _total_scope(rows)
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < exp(-delta / temp):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
            else:
                # revert
                row[elem] = old_val
                for d in new_changed:
                    c = counter[d]
                    if c == 1:
                        bad += 1
                    elif c == 2:
                        bad -= 1
                    counter[d] = c - 1
                    if counter[d] == 0:
                        del counter[d]
                for d in changed:
                    c = counter.get(d, 0)
                    if c == 1:
                        bad += 1
                    elif c == 0:
                        bad -= 1
                    counter[d] = c + 1
        else:
            # invalid: revert
            row[elem] = old_val
            for d in new_changed:
                c = counter[d]
                if c == 1:
                    bad += 1
                elif c == 2:
                    bad -= 1
                counter[d] = c - 1
                if counter[d] == 0:
                    del counter[d]
            for d in changed:
                c = counter.get(d, 0)
                if c == 1:
                    bad += 1
                elif c == 0:
                    bad -= 1
                counter[d] = c + 1

        temp *= cool

    return best_rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)

    rows = _greedy_construct(n_rows, k)
    if rows is None:
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

    deadline = time.time() + 138  # safety margin well under the 150s kill
    best_rows = rows
    best_scope = _total_scope(rows)

    # Phase 1: randomized greedy restarts (diverse starting points).
    for _ in range(25):
        if time.time() > deadline:
            break
        rand_rows = _randomized_greedy(n_rows, k, rng, pool_size=rng.choice([3, 5, 8]))
        if rand_rows is not None:
            scope = _total_scope(rand_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rand_rows]

    # Phase 2: deterministic local search (quick descent).
    if time.time() < deadline - 20:
        ls_rows = _local_search(best_rows, k, rng, max_iterations=4000)
        scope = _total_scope(ls_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in ls_rows]

    # Phase 3: simulated annealing (escape local optima), repeated.
    if time.time() < deadline - 5:
        for _ in range(4):
            if time.time() > deadline - 2:
                break
            sa_rows = _sa_search(best_rows, k, rng, deadline - 2)
            scope = _total_scope(sa_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in sa_rows]

    return best_rows

# EVOLVE-BLOCK-END
