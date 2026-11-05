"""Difference triangle set (7,5) — joint simulated-annealing search over all rows."""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of positive differences for a single row (list of k+1 ints)."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the full set of differences across all rows."""
    s = set()
    for row in rows:
        s |= _row_diffs(row)
    return s


def _valid(rows):
    """Check that all positive differences are distinct over all rows."""
    total = 0
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _scope(rows):
    return max(max(row) for row in rows)


def _greedy_construct(n_rows, k):
    """Build rows greedily (same as seed) to get a valid starting point."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _random_valid_row(k, used, max_val):
    """Try to build a random valid row given already-used differences."""
    for _attempt in range(50):
        row = [0]
        x = 0
        ok = True
        while len(row) < k + 1:
            # Pick a random increment
            x += random.randint(1, 3)
            new = {x - a for a in row}
            if new & used:
                ok = False
                break
            row.append(x)
            used_local = used | new
        if ok and max(row) <= max_val:
            return row
    return None


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Initial solution: greedy
    rows = _greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)

    # Simulated annealing
    n_marks = k + 1  # marks per row
    total_diffs = n_rows * (k * (k + 1) // 2)  # 7 * 15 = 105

    # Temperature schedule
    t0 = 50.0
    t_end = 0.1
    # We'll run iterations until deadline; use a geometric cooling
    # Estimate number of iterations we can afford
    # Each iteration is O(n_rows * k^2) roughly

    current_scope = best_scope
    temp = t0

    # Precompute: for efficiency, track used diffs as a set
    used_diffs = _all_diffs(rows)

    rng = random.Random(42)

    iteration = 0
    while time.time() < deadline:
        iteration += 1

        # Pick a random row and a random mark to move
        r_idx = rng.randrange(n_rows)
        m_idx = rng.randrange(1, n_marks)  # don't move the 0 mark

        old_val = rows[r_idx][m_idx]
        # Propose a new value: random perturbation
        delta = rng.choice([-3, -2, -1, 1, 2, 3, -5, 5])
        new_val = old_val + delta

        # Constraints: must keep row strictly increasing
        if new_val <= rows[r_idx][m_idx - 1]:
            new_val = rows[r_idx][m_idx - 1] + 1
        if m_idx < n_marks - 1 and new_val >= rows[r_idx][m_idx + 1]:
            new_val = rows[r_idx][m_idx + 1] - 1
        if new_val <= rows[r_idx][m_idx - 1]:
            continue  # can't fix, skip

        # Compute old diffs for this row involving m_idx
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(n_marks):
            if i != m_idx:
                d = abs(old_row[m_idx] - old_row[i])
                if d > 0:
                    old_diffs.add(d)

        # Compute new diffs for this row involving m_idx
        new_diffs = set()
        for i in range(n_marks):
            if i != m_idx:
                d = abs(new_val - old_row[i])
                if d > 0:
                    new_diffs.add(d)

        # Check: new_diffs must not conflict with used_diffs minus old_diffs
        other_diffs = used_diffs - old_diffs
        if new_diffs & other_diffs:
            # Conflict: reject
            continue

        # Accept or reject based on annealing
        new_scope = max(max(rows[r_idx]) if False else max(old_row[:m_idx] + [new_val] + old_row[m_idx+1:]),
                        max(max(r) for r in rows if r is not rows[r_idx])) if n_rows > 1 else max(old_row[:m_idx] + [new_val] + old_row[m_idx+1:])

        # Actually compute scope properly
        new_row = old_row[:m_idx] + [new_val] + old_row[m_idx+1:]
        new_row_scope = max(new_row)
        other_max = max(max(r) for r in rows if r is not rows[r_idx]) if n_rows > 1 else 0
        new_scope_val = max(new_row_scope, other_max)

        delta_scope = new_scope_val - current_scope

        if delta_scope <= 0:
            # Improvement or equal: accept
            accept = True
        else:
            # Worse: accept with probability exp(-delta_scope / temp)
            if temp < 1e-9:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta_scope / temp)

        if accept:
            # Apply move
            rows[r_idx][m_idx] = new_val
            used_diffs = (used_diffs - old_diffs) | new_diffs
            current_scope = new_scope_val

            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]

        # Cool down
        temp *= 0.99995

        # Periodic restart if stuck for too long
        if iteration % 5000 == 0 and temp < t_end:
            # Try to reheat slightly
            temp = max(temp, 1.0)

    return best_rows

# EVOLVE-BLOCK-END
