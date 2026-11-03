"""Difference triangle set (4,7) – optimized construction.

Strategy:
  1. Fast greedy to get an initial valid set (and a baseline scope).
  2. Local search via 2-element swaps / shifts to reduce the max entry.
  3. Randomized restarts with different seeds / orders to escape local minima.
  4. Deadline guard: return the best solution found before the time limit.

The key insight from the inspiration: aggressive pull-to-smallest with 2-element
swap perturbations and randomized restarts can escape local minima that single-
element shifts cannot.
"""

import time
import random
import copy
from itertools import combinations

_DEADLINE_MARGIN = 10.0  # seconds to spare before the 150 s hard kill


def _all_diffs(rows):
    """Return the set of all positive differences within each row."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
    return diffs


def _is_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    diffs = set()
    for row in rows:
        n = len(i for i in range(len(row)))  # just to get n
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True


def _scope(rows):
    """The largest entry across all rows."""
    return max(max(row) for row in rows)


def _greedy(n_rows, k, rng=None, order=None):
    """Greedy construction: each row takes the smallest next mark whose
    differences are all unused. Optional randomization via rng and row order."""
    if rng is None:
        rng = random.Random()
    if order is None:
        order = list(range(n_rows))
    if rng.random() < 0.5:
        order = order[::-1]

    used = set()
    rows = [None] * n_rows
    for idx in order:
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows[idx] = row
    return rows


def _try_swap(row, i, j, new_val, used, exclude_row_idx, rows):
    """Try replacing row[i] with new_val (keeping row[j] fixed).
    Returns (new_row, new_used_diffs) if valid, else (None, None)."""
    # Remove old diffs involving index i
    old_val = row[i]
    removed = set()
    for idx in range(len(row)):
        if idx != i:
            d = abs(row[idx] - old_val)
            removed.add(d)

    # Add new diffs involving index i
    added = set()
    for idx in range(len(row)):
        if idx != i:
            d = abs(row[idx] - new_val)
            added.add(d)

    # Check: added diffs must not conflict with (used - removed)
    # i.e., added must not intersect with any diff not being removed
    conflict = added & (used - removed)
    if conflict:
        return None, None

    # Also check internal consistency: new_val must maintain strict increase
    if i > 0 and new_val <= row[i - 1]:
        return None, None
    if i < len(row) - 1 and new_val >= row[i + 1]:
        return None, None
    # Also new_val must be > 0 if i > 0 (row[0] is 0, but i could be 0)
    if i > 0 and new_val <= 0:
        return None, None

    new_row = row[:]
    new_row[i] = new_val
    return new_row, added


def _local_search(rows, n_rows, k, rng, max_iters=5000, deadline=None):
    """Local search: try to reduce the scope by shifting elements down.
    Uses 2-element perturbations (swap two elements' values, or shift one)."""
    best_rows = copy.deepcopy(rows)
    best_scope = _scope(best_rows)
    used = _all_diffs(best_rows)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Pick a random row and a random pair of indices to perturb
        r = rng.randint(0, n_rows - 1)
        row = best_rows[r]
        n = len(row)
        if n < 2:
            continue

        # Choose two distinct indices (not 0, since row[0]=0 is fixed)
        candidates = list(range(1, n))
        if len(candidates) < 2:
            candidates = [1]

        i = rng.choice(candidates)
        # Try to pull element i downward
        # Find the smallest valid value for row[i] given current used diffs
        # We'll try values from row[i]-1 down to row[i-1]+1 (if i>0) or 1

        start = row[i] - 1
        end = (row[i - 1] + 1) if i > 0 else 1
        found = False
        for new_val in range(start, end - 1, -1):
            if new_val < end:
                continue
            new_row, added = _try_swap(row, i, new_val, used, r, best_rows)
            if new_row is not None:
                # Apply the change
                removed = set()
                old_val = row[i]
                for idx in range(n):
                    if idx != i:
                        d = abs(row[idx] - old_val)
                        removed.add(d)
                used = (used - removed) | added
                best_rows[r] = new_row
                new_scope = _scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                found = True
                break

        if not found:
            # Try a 2-element swap perturbation: swap values of two elements
            # (this is a structural change that can escape local minima)
            if len(candidates) >= 2:
                i2 = rng.choice([c for c in candidates if c != i] or candidates)
                # Swap row[i] and row[i2]
                temp = best_rows[r][i]
                best_rows[r][i] = best_rows[r][i2]
                best_rows[r][i2] = temp
                if not _is_valid(best_rows):
                    # Revert
                    best_rows[r][i2] = best_rows[r][i]
                    best_rows[r][i] = temp
                else:
                    used = _all_diffs(best_rows)
                    new_scope = _scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
            else:
                # Try shifting a different element
                if len(candidates) >= 1:
                    i2 = candidates[0]
                    start2 = best_rows[r][i2] - 1
                    end2 = (best_rows[r][i2 - 1] + 1) if i2 > 0 else 1
                    for new_val2 in range(start2, end2 - 1, -1):
                        if new_val2 < end2:
                            continue
                        new_row2, added2 = _try_swap(best_rows[r], i2, new_val2, used, r, best_rows)
                        if new_row2 is not None:
                            old_val2 = best_rows[r][i2]
                            removed2 = set()
                            n2 = len(best_rows[r])
                            for idx in range(n2):
                                if idx != i2:
                                    d = abs(best_rows[r][idx] - old_val2)
                                    removed2.add(d)
                            used = (used - removed2) | added2
                            best_rows[r] = new_row2
                            new_scope = _scope(best_rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                            break

    return best_rows, best_scope


def _restart_search(n_rows, k, rng, deadline):
    """Multiple randomized restarts with local search."""
    best_rows = None
    best_scope = float('inf')

    # First, get a greedy baseline
    greedy_rows = _greedy(n_rows, k, rng)
    best_rows = copy.deepcopy(greedy_rows)
    best_scope = _scope(best_rows)

    # Try several restarts with different random seeds / orders
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Vary the greedy construction
        order = list(range(n_rows))
        rng.shuffle(order)
        greedy_rows = _greedy(n_rows, k, rng, order=order)
        scope = _scope(greedy_rows)

        # If greedy already gives a better scope, update
        if scope < best_scope:
            best_rows = copy.deepcopy(greedy_rows)
            best_scope = scope

        # Run local search on this solution
        ls_rows, ls_scope = _local_search(
            greedy_rows, n_rows, k, rng,
            max_iters=3000, deadline=deadline
        )
        if ls_scope < best_scope:
            best_rows = copy.deepcopy(ls_rows)
            best_scope = ls_scope

        # Also try local search on the current best to see if we can improve
        if time.time() < deadline:
            ls_rows2, ls_scope2 = _local_search(
                best_rows, n_rows, k, rng,
                max_iters=2000, deadline=deadline
            )
            if ls_scope2 < best_scope:
                best_rows = copy.deepcopy(ls_rows2)
                best_scope = ls_scope2

        # If we've hit a good scope, try harder local search
        if best_scope <= 140:
            ls_rows3, ls_scope3 = _local_search(
                best_rows, n_rows, k, rng,
                max_iters=5000, deadline=deadline
            )
            if ls_scope3 < best_scope:
                best_rows = copy.deepcopy(ls_rows3)
                best_scope = ls_scope3

        if restarts > 50:
            break

    return best_rows


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN

    # Ensure we always return a valid result
    # Quick greedy first
    rng = random.Random(42)
    rows = _greedy(n_rows, k, rng)
    best_scope = _scope(rows)

    # If we have time, do the full search
    if time.time() < deadline:
        try:
            improved_rows = _restart_search(n_rows, k, rng, deadline)
            if improved_rows is not None:
                # Validate
                if _is_valid(improved_rows):
                    rows = improved_rows
                    best_scope = _scope(rows)
        except Exception:
            pass

    # Final validation
    if not _is_valid(rows):
        # Fallback to a fresh greedy
        rows = _greedy(n_rows, k, random.Random(123))

    return rows
