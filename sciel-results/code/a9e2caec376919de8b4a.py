"""Difference triangle set (4,7): construct rows with distinct differences, minimizing scope."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # The number of differences per row is k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on scope: we need at least total_diffs distinct positive differences
    # The largest difference in a row starting at 0 is the last element.
    # If scope is S, the max difference is S, so we need S >= total_diffs (since diffs are 1..S)
    # Actually differences can be any values up to S, so we need at least total_diffs distinct values in [1, S]
    # Thus S >= total_diffs
    lb = total_diffs

    best_result = None
    best_scope = float('inf')

    # Try to build with greedy + local search approach
    # First, get a valid solution using greedy
    greedy_rows = _greedy_construct(n_rows, k)
    greedy_scope = max(max(r) for r in greedy_rows)
    best_result = greedy_rows
    best_scope = greedy_scope

    if time.time() > deadline:
        return best_result

    # Try to improve scope using iterative local search
    # Strategy: try to reduce scope by re-optimizing rows
    # Use a simulated annealing / hill climbing approach on the set of rows

    for attempt in range(200):
        if time.time() > deadline - 5:
            break

        # Random restart with perturbation
        current_rows = _perturb(greedy_rows, n_rows, k)
        current_scope = max(max(r) for r in current_rows)

        if current_scope < best_scope:
            best_scope = current_scope
            best_result = [list(r) for r in current_rows]

        # Local search: try to reduce scope by adjusting elements
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try each row, each position, try to decrease the value
            for ri in range(n_rows):
                for pos in range(k + 1):
                    if pos == 0:
                        continue
                    orig_val = current_rows[ri][pos]
                    # Try to decrease this value
                    for new_val in range(orig_val - 1, current_rows[ri][pos - 1], -1):
                        if time.time() > deadline - 3:
                            break
                        # Check if setting current_rows[ri][pos] = new_val is valid
                        # We need to check:
                        # 1. Strictly increasing: new_val > current_rows[ri][pos-1] (guaranteed by range)
                        # 2. All differences in this row are distinct from all other rows' differences
                        #    and distinct within this row
                        trial_row = current_rows[ri][:]
                        trial_row[pos] = new_val
                        # Check increasing
                        if any(trial_row[j] >= trial_row[j+1] for j in range(len(trial_row)-1)):
                            continue
                        # Check all differences are globally distinct
                        if _check_validity(current_rows, ri, trial_row):
                            current_rows[ri][pos] = new_val
                            current_scope = max(max(r) for r in current_rows)
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_result = [list(r) for r in current_rows]
                            improved = True
                            break

    return best_result


def _greedy_construct(n_rows, k):
    """Greedy construction: build each row using smallest available differences."""
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


def _perturb(rows, n_rows, k):
    """Create a perturbed copy of rows for local search."""
    perturbed = [list(r) for r in rows]
    # Randomly add some offset to non-first elements to create variation
    for ri in range(n_rows):
        for pos in range(1, len(perturbed[ri])):
            if random.random() < 0.3:
                # Add a small random offset, maintaining increasing order
                max_add = 5
                if pos < len(perturbed[ri]) - 1:
                    room = perturbed[ri][pos + 1] - perturbed[ri][pos] - 1
                    max_add = min(max_add, room)
                if max_add > 0:
                    add = random.randint(1, max_add)
                    perturbed[ri][pos] += add
                    # Ensure subsequent elements are still increasing
                    for j in range(pos + 1, len(perturbed[ri])):
                        if perturbed[ri][j] <= perturbed[ri][j-1]:
                            perturbed[ri][j] = perturbed[ri][j-1] + 1
    return perturbed


def _check_validity(rows, changed_row_idx, trial_row):
    """
    Check if replacing rows[changed_row_idx] with trial_row maintains global distinctness.
    Returns True if valid.
    """
    # Collect all differences from other rows
    other_diffs = set()
    for i, row in enumerate(rows):
        if i == changed_row_idx:
            continue
        for j in range(len(row)):
            for l in range(j + 1, len(row)):
                other_diffs.add(row[l] - row[j])

    # Check trial_row's differences
    trial_diffs = set()
    for j in range(len(trial_row)):
        for l in range(j + 1, len(trial_row)):
            d = trial_row[l] - trial_row[j]
            if d in other_diffs:
                return False
            if d in trial_diffs:
                return False
            trial_diffs.add(d)

    return True
