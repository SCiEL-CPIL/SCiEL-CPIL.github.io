import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0

    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1

    k_plus_1 = k + 1

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Polish pass for further compaction
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Try to find better solutions with randomized backtracking
    attempt = 0
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        # Try multiple random restarts for this target scope
        found = False
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Try to improve further
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    # Polish
                    polished = _polish_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in polished)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = polished
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 5:
            # If we can't find anything near the lower bound, stop
            if target_scope < lower_bound:
                break

    # Final polish
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    return best_solution

def greedy_construct(n_rows, k):
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

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses bitmask for fast difference checking.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1

    # Use a bitmask for used differences
    used_mask = 0  # Python int, arbitrary precision

    def build_row(pos, current_mark, row_so_far, mask):
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        # Find valid candidates
        valid_candidates = []
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > max_scope:
                    conflict = True
                    break
                if (mask >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)

        if not valid_candidates:
            return None

        n_smallest = min(15, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]

        if len(valid_candidates) > n_smallest and rng.random() < 0.25:
            extra = rng.sample(valid_candidates[n_smallest:], min(8, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            new_mask = mask
            for a in row_so_far:
                d = x - a
                new_mask |= (1 << d)

            res = build_row(pos + 1, x, row_so_far + [x], new_mask)
            if res is not None:
                return res

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0], used_mask)
        if row is None:
            return None
        rows.append(row)
        # Update mask with all differences in this row
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                used_mask |= (1 << d)

    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope of a valid solution by local search.
    Uses a "pull to smallest" strategy.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def _pull_to_smallest(row_idx):
        row = best[row_idx]
        any_changed = False
        for pos in range(1, k_plus_1):
            original = row[pos]
            # Try to find the smallest value > row[pos-1] that is valid
            for new_val in range(row[pos - 1] + 1, original + 1):
                if new_val == original:
                    break
                
                # Check conflicts
                new_diffs = set()
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0:
                        break
                    new_diffs.add(d)
                else:
                    ok = True
                    for j in range(pos + 1, k_plus_1):
                        d = row[j] - new_val
                        if d <= 0:
                            ok = False
                            break
                        new_diffs.add(d)
                    
                    if not ok:
                        continue

                    # Check if new_diffs conflict with used_diffs (excluding old diffs of this element)
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)
                    
                    # The set of diffs that must be free is (new_diffs - old_diffs)
                    # But we must also ensure new_diffs doesn't contain any diff that is in used_diffs but NOT in old_diffs
                    conflict = False
                    for d in new_diffs:
                        if d in used_diffs and d not in old_diffs:
                            conflict = True
                            break
                    
                    if conflict:
                        continue

                    # If no conflict, apply
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_changed = True
                    break

        return any_changed

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                improved = True

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]

def _polish_scope(solution, n_rows, k, deadline):
    """
    Aggressive polishing: repeatedly try single-element shifts to minimize scope.
    This is a tighter local search that can find small improvements.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    # Iteratively try to reduce each element
    for iteration in range(20):
        if time.time() > deadline - 2.0:
            break

        any_improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                # Try shifting down by 1, 2, ... up to original - row[pos-1] - 1
                for new_val in range(original - 1, row[pos - 1], -1):
                    # Check if new value creates conflicts
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)

                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)

                    # Check if new_diffs conflict with (used_diffs - old_diffs)
                    if new_diffs & (used_diffs - old_diffs):
                        continue

                    # Check if new_diffs are all positive
                    if any(d <= 0 for d in new_diffs):
                        continue

                    # Apply the change
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_improved = True
                    break

        if not any_improved:
            break

    return best
