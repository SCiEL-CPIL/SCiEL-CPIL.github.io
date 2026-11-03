import time
import random
import sys
from functools import lru_cache
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Bitmask-based backtracking search to reduce scope
    if time.time() < deadline - 5.0:
        attempt = 0
        while time.time() < deadline - 3.0:
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    # Polish with local search
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
                else:
                    break
            attempt += 1

    # 3. Final local improvement pass
    if time.time() < deadline - 1.0:
        improved = _improve_scope(best_solution, n_rows, k, deadline)
        new_scope = max(max(row) for row in improved)
        if new_scope < best_scope:
            best_scope = new_scope
            best_solution = improved

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
    Bitmask-based backtracking search.
    
    Structural change: instead of Python sets for difference tracking,
    we precompute a bitmask for each candidate value x given the current
    partial row. A conflict check is a single bitwise AND (O(1) in word size).
    
    We build rows one at a time. For each row, we backtrack over choices
    for each position, using the bitmask to quickly reject conflicts.
    """
    rng = random.Random(seed_offset * 104729 + 12345)
    k_plus_1 = k + 1
    total_diffs_needed = n_rows * k * (k + 1) // 2
    # max_scope must be at least total_diffs_needed - 1 for feasibility
    if max_scope < total_diffs_needed - 1:
        return None

    # We'll use an integer bitmask to track used differences.
    # Difference d is tracked at bit position (d-1).
    # max_scope differences means we need bits 0..max_scope-1
    # Use a Python int as an arbitrary-precision bitmask.

    used_mask = 0  # bitmask of used differences
    rows = []

    def row_diff_mask(row, pos):
        """Compute the bitmask of differences introduced by adding row[pos] to row[:pos]."""
        x = row[pos]
        mask = 0
        for i in range(pos):
            d = x - row[i]
            if 1 <= d <= max_scope:
                mask |= (1 << (d - 1))
        return mask

    def build_row(row_idx):
        """Recursively build one row using backtracking."""
        nonlocal used_mask
        if time.time() > deadline:
            return False

        row = [0]
        pos = 1

        def backtrack(pos, row):
            nonlocal used_mask
            if time.time() > deadline:
                return False

            if pos == k_plus_1:
                rows.append(list(row))
                return True

            last_val = row[-1]
            # Maximum x we can place: need room for remaining elements
            remaining = k_plus_1 - pos
            max_x = max_scope - (remaining - 1)
            if max_x <= last_val:
                return False

            # Generate candidates
            candidates = []
            for x in range(last_val + 1, max_x + 1):
                # Compute diff mask for this candidate
                mask = 0
                conflict = False
                for i in range(pos):
                    d = x - row[i]
                    if d <= 0 or d > max_scope:
                        conflict = True
                        break
                    mask |= (1 << (d - 1))
                if conflict:
                    continue
                # Check against used differences
                if mask & used_mask:
                    continue
                candidates.append((x, mask))

            if not candidates:
                return False

            # Order candidates
            if seed_offset % 4 == 0:
                candidates.sort(key=lambda t: t[0])
            elif seed_offset % 4 == 1:
                rng.shuffle(candidates)
            elif seed_offset % 4 == 2:
                # Prefer candidates that use "spread out" differences
                candidates.sort(key=lambda t: (t[0], -t[1].bit_count()))
            else:
                candidates.sort(key=lambda t: t[0])
                # Add some randomness at the end
                if len(candidates) > 1:
                    rng.shuffle(candidates[-min(3, len(candidates)):])

            # Limit branching for deep positions
            max_branch = 20
            if len(candidates) > max_branch:
                # Keep smallest and some random
                keep = candidates[:max_branch - 2]
                if len(candidates) > max_branch - 2:
                    rest = candidates[max_branch - 2:]
                    keep += rng.sample(rest, min(2, len(rest)))
                candidates = keep

            for x, mask in candidates:
                if time.time() > deadline:
                    return False

                # Apply
                old_mask = used_mask
                used_mask |= mask
                row.append(x)

                if backtrack(pos + 1, row):
                    return True

                # Undo
                row.pop()
                used_mask = old_mask

            return False

        return backtrack(pos, row)

    # Build all rows
    for r in range(n_rows):
        if not build_row(r):
            return None

    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Local search to reduce scope by shifting elements down.
    Uses bitmask for O(1) conflict checks.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Build initial used-difference set
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                # Try shifting down
                for new_val in range(row[pos - 1] + 1, original):
                    # Compute new diffs
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)

                    # Compute old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)

                    # Check conflicts
                    if new_diffs & (used_diffs - old_diffs):
                        continue
                    if any(d <= 0 for d in new_diffs):
                        continue

                    # Apply
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
