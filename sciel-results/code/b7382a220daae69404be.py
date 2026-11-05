"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Find a difference triangle set: n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin out of 150s

    # Total number of differences per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on scope
    lb = total_diffs

    # Best solution found so far
    best_scope = None
    best_rows = None

    # Greedy baseline first (fast, gives a decent upper bound)
    greedy_rows = _greedy_construct(n_rows, k)
    greedy_scope = max(max(row) for row in greedy_rows)
    best_scope = greedy_scope
    best_rows = greedy_rows

    rng = random.Random(42)

    # Try to find better solutions using randomized backtracking
    for target_scope in range(best_scope - 1, max(lb - 1, 0), -1):
        if time.time() > deadline:
            break

        found = False
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            result = _solve_with_scope(n_rows, k, target_scope, rng, deadline)
            if result is not None:
                scope = max(max(row) for row in result)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = result
                found = True
                break

        if found:
            continue
        else:
            if target_scope - lb < 5:
                break

    return best_rows


def _greedy_construct(n_rows, k):
    """Standard greedy construction."""
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


def _solve_with_scope(n_rows, k, max_scope, rng, deadline):
    """
    Try to find a valid difference triangle set with all entries <= max_scope
    using randomized backtracking with bitmask for difference tracking.
    """
    state = {'used_mask': 0, 'rows': []}

    def backtrack(row_idx, row):
        if time.time() > deadline:
            return False

        if row_idx == n_rows:
            return True

        if len(row) == k + 1:
            state['rows'].append(row[:])
            return backtrack(row_idx + 1, [0])

        start_x = row[-1] + 1
        end_x = max_scope

        candidates = []
        for x_val in range(start_x, end_x + 1):
            new_mask = 0
            valid = True
            for a in row:
                d = x_val - a
                if d > max_scope:
                    valid = False
                    break
                bit = 1 << d
                if state['used_mask'] & bit:
                    valid = False
                    break
                new_mask |= bit

            if valid:
                candidates.append((x_val, new_mask))

            if len(candidates) >= 20:
                break

        if not candidates:
            return False

        if len(candidates) > 1:
            rng.shuffle(candidates)

        for x_val, new_mask in candidates:
            old_mask = state['used_mask']
            state['used_mask'] = old_mask | new_mask
            row.append(x_val)

            if backtrack(row_idx, row):
                return True

            row.pop()
            state['used_mask'] = old_mask

        return False

    if backtrack(0, [0]):
        return state['rows']
    return None
# EVOLVE-BLOCK-END
