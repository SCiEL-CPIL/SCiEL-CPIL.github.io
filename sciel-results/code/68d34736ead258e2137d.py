"""Difference triangle set (4,7) solver with DFS + bitset pruning and deadline guard."""
import time
from itertools import combinations

# Deadline: evaluator kills at 150s; keep a safety margin.
_DEADLINE = 135.0

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # For the specific task: n_rows=4, k=7 (8 entries per row)
    # But we handle general case.

    # Number of differences per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Greedy baseline: build rows greedily
    best_rows = _greedy(n_rows, k)
    best_scope = max(max(r) for r in best_rows)

    # Try to improve via DFS with backtracking
    # We'll try to find a solution with scope < best_scope
    # Use iterative deepening on the maximum value

    # Lower bound: counting bound
    # total_diffs distinct positive integers, max diff <= scope
    # Number of positive integers <= scope is scope
    # So we need scope >= total_diffs
    lb = total_diffs
    # Actually the counting bound is: we need total_diffs distinct values in [1, scope]
    # so scope >= total_diffs

    # Try DFS from lb upward
    for max_val in range(lb, best_scope):
        if time.time() > deadline:
            break
        result = _dfs_search(n_rows, k, max_val)
        if result is not None:
            best_rows = result
            best_scope = max_val
            # We found a solution with this scope; try to go lower
            # But since we're going upward from lb, the first found is the best
            # Actually we should search downward from best_scope-1
            break

    # Try DFS downward from current best
    for max_val in range(best_scope - 1, lb - 1, -1):
        if time.time() > deadline:
            break
        result = _dfs_search(n_rows, k, max_val)
        if result is not None:
            best_rows = result
            best_scope = max_val

    return best_rows


def _greedy(n_rows, k):
    """Greedy construction: each row takes smallest next mark with unused differences."""
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


def _dfs_search(n_rows, k, max_val):
    """
    DFS to find n_rows rows of k+1 integers in [0, max_val] with distinct differences.
    Returns list of rows or None.
    """
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Use a bitset for used differences (1..max_val)
    # max_val can be up to ~134, so bitset of 135 bits fits in an int
    used_mask = 0
    rows = []

    def dfs(row_idx, pos_in_row, current_row):
        nonlocal used_mask

        if time.time() > _DEADLINE:
            return False

        if pos_in_row == k + 1:
            # Row complete
            rows.append(list(current_row))
            if row_idx == n_rows - 1:
                return True
            return dfs(row_idx + 1, 0, [0])

        if pos_in_row == 0:
            # First element is always 0
            return dfs(row_idx, 1, [0])

        # Try next values
        prev = current_row[-1]
        start_val = prev + 1

        for val in range(start_val, max_val + 1):
            # Check if adding val creates any used difference
            conflict = False
            for a in current_row:
                d = val - a
                if d > 0 and (used_mask >> d) & 1:
                    conflict = True
                    break
            if conflict:
                continue

            # Also check: all differences among new set must be distinct within this row
            # Since we're building incrementally and checking against used_mask which
            # includes all previously used diffs, and we haven't added this row's diffs yet,
            # we need to ensure no two diffs in this row collide.
            # Actually, we should check that the new diffs don't collide with each other
            # or with used_mask.
            new_diffs = [val - a for a in current_row]
            # Check for duplicates within new_diffs
            if len(set(new_diffs)) != len(new_diffs):
                continue
            # Check against used_mask
            ok = True
            for d in new_diffs:
                if (used_mask >> d) & 1:
                    ok = False
                    break
            if not ok:
                continue

            # Accept
            new_mask = used_mask
            for d in new_diffs:
                new_mask |= (1 << d)

            current_row.append(val)
            if dfs(row_idx, pos_in_row + 1, current_row):
                return True
            current_row.pop()
            used_mask = new_mask  # restore

        return False

    # Symmetry breaking: first row should start with 0,1,... 
    # Actually, to reduce search, we can fix the first row's first few elements.
    # But for now, let's just run DFS.

    # To speed up, we can use the fact that the first element is 0.
    # Let's try to build rows one at a time.

    # Actually, the above DFS has a bug: when we backtrack, we need to restore used_mask correctly.
    # Let me rewrite more carefully.

    used_mask = 0
    rows = []

    def dfs2(row_idx, pos_in_row, current_row):
        nonlocal used_mask

        if time.time() > _DEADLINE:
            return False

        if pos_in_row == k + 1:
            rows.append(list(current_row))
            if row_idx == n_rows - 1:
                return True
            return dfs2(row_idx + 1, 0, [0])

        if pos_in_row == 0:
            return dfs2(row_idx, 1, [0])

        prev = current_row[-1]
        for val in range(prev + 1, max_val + 1):
            new_diffs = [val - a for a in current_row]
            # Check uniqueness within new_diffs
            if len(set(new_diffs)) != len(new_diffs):
                continue
            # Check against used_mask
            ok = True
            for d in new_diffs:
                if (used_mask >> d) & 1:
                    ok = False
                    break
            if not ok:
                continue

            old_mask = used_mask
            for d in new_diffs:
                used_mask |= (1 << d)

            current_row.append(val)
            if dfs2(row_idx, pos_in_row + 1, current_row):
                return True
            current_row.pop()
            used_mask = old_mask

        return False

    if dfs2(0, 0, []):
        return rows
    return None
