import time
import random
import sys
import numpy as np
from collections import defaultdict
from itertools import combinations
import heapq

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 117.0  # safety margin before 120s

    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Aggressive local improvement
    best_solution = _aggressive_local_search(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Multi-target backtracking: try to find solutions with scope = best_scope-1, -2, ...
    if time.time() < deadline - 8.0:
        for delta in range(1, 15):
            if time.time() > deadline - 8.0:
                break
            target = best_scope - delta
            if target < lower_bound:
                break
            # Try multiple random restarts
            found = False
            for attempt in range(20):
                if time.time() > deadline - 8.0:
                    break
                sol = _backtrack_with_bitset(n_rows, k, target, deadline, seed_offset=attempt * 97 + 13)
                if sol:
                    new_scope = max(max(row) for row in sol)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = sol
                        found = True
                        # Polish the found solution
                        best_solution = _aggressive_local_search(best_solution, n_rows, k, deadline)
                        new_scope = max(max(row) for row in best_solution)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_solution = best_solution
                        if best_scope <= lower_bound:
                            return best_solution
                        break
            if best_scope <= lower_bound:
                return best_solution

    # 4. Additional polish passes
    if time.time() < deadline - 2.0:
        best_solution = _polish_with_bitset(best_solution, n_rows, k, deadline)
        new_scope = max(max(row) for row in best_solution)
        if new_scope < best_scope:
            best_scope = new_scope
            best_solution = best_solution

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

def _aggressive_local_search(solution, n_rows, k, deadline):
    """
    Multi-strategy local search:
    1. Pull-to-smallest-feasible (single element moves)
    2. Two-element swap/shift moves
    3. Row reordering to find better packings
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    k_plus_1 = k + 1
    best_scope = max(max(row) for row in best)

    # Build used_diffs set
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    improved = True
    passes = 0
    while improved and time.time() < deadline and passes < 50:
        improved = False
        passes += 1

        # Pass 1: Pull each element to smallest feasible
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                for new_val in range(row[pos - 1] + 1, original):
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)

                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)

                    if any(d <= 0 for d in new_diffs):
                        continue
                    if new_diffs & (used_diffs - old_diffs):
                        continue
                    if len(new_diffs) != len(list(new_diffs)):
                        continue

                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break

        # Pass 2: Try two-element moves (shift both down or up)
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos1 in range(1, k_plus_1):
                for pos2 in range(pos1 + 1, k_plus_1):
                    orig1 = row[pos1]
                    orig2 = row[pos2]
                    # Try shifting both down by 1
                    new1 = orig1 - 1
                    new2 = orig2 - 1
                    if new1 <= row[pos1 - 1] or new2 <= row[pos2 - 1] or new2 <= new1:
                        continue

                    new_diffs = set()
                    for i in range(pos1):
                        new_diffs.add(new1 - row[i])
                    for j in range(pos1 + 1, k_plus_1):
                        if j != pos2:
                            new_diffs.add(row[j] - new1)
                        else:
                            new_diffs.add(new2 - new1)
                    for i in range(pos2):
                        if i != pos1:
                            new_diffs.add(new2 - row[i])
                        else:
                            pass  # already handled
                    for j in range(pos2 + 1, k_plus_1):
                        new_diffs.add(row[j] - new2)

                    old_diffs = set()
                    for i in range(pos1):
                        old_diffs.add(orig1 - row[i])
                    for j in range(pos1 + 1, k_plus_1):
                        if j != pos2:
                            old_diffs.add(row[j] - orig1)
                        else:
                            old_diffs.add(orig2 - orig1)
                    for i in range(pos2):
                        if i != pos1:
                            old_diffs.add(orig2 - row[i])
                    for j in range(pos2 + 1, k_plus_1):
                        old_diffs.add(row[j] - orig2)

                    if any(d <= 0 for d in new_diffs):
                        continue
                    if new_diffs & (used_diffs - old_diffs):
                        continue

                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos1] = new1
                    row[pos2] = new2
                    improved = True
                    break
                if improved:
                    break

    return best

def _polish_with_bitset(solution, n_rows, k, deadline):
    """
    High-precision polish using bitset for O(1) difference lookup.
    Multiple passes of pull-to-smallest-feasible.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    k_plus_1 = k + 1
    best_scope = max(max(row) for row in best)

    # Bitset: bit d set means difference d is used
    max_d = best_scope + k
    used_bits = 0

    def is_used(d):
        return (used_bits >> d) & 1

    def mark(d):
        nonlocal used_bits
        used_bits |= (1 << d)

    def unmark(d):
        nonlocal used_bits
        used_bits &= ~(1 << d)

    # Initialize
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                mark(row[j] - row[i])

    for _pass in range(30):
        if time.time() > deadline:
            break
        changed = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                # Collect old diffs
                old_diffs = []
                for i in range(pos):
                    old_diffs.append(original - row[i])
                for j in range(pos + 1, k_plus_1):
                    old_diffs.append(row[j] - original)

                for new_val in range(row[pos - 1] + 1, original):
                    new_diffs = []
                    for i in range(pos):
                        new_diffs.append(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.append(row[j] - new_val)

                    if any(d <= 0 for d in new_diffs):
                        break
                    # Check conflicts: new_diffs not in (used_bits - old_diffs)
                    conflict = False
                    old_set = set(old_diffs)
                    for d in new_diffs:
                        if is_used(d) and d not in old_set:
                            conflict = True
                            break
                    if conflict:
                        continue
                    # Check internal duplicates
                    if len(set(new_diffs)) != len(new_diffs):
                        continue

                    # Apply
                    for d in old_diffs:
                        unmark(d)
                    for d in new_diffs:
                        mark(d)
                    row[pos] = new_val
                    changed = True
                    break

        if not changed:
            break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    return best

def _backtrack_with_bitset(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search with bitset for O(1) difference checking.
    Tries to build n_rows rows each with k+1 elements, all in [0, max_scope].
    """
    rng = random.Random(seed_offset * 7919 + 31)
    k_plus_1 = k + 1
    used_bits = 0
    rows = []

    def build_row(start_pos, last_val, row_so_far):
        nonlocal used_bits
        if time.time() > deadline:
            return None

        if start_pos == k_plus_1:
            return list(row_so_far)

        # Upper bound: need (k_plus_1 - start_pos - 1) more elements,
        # each at least 1 greater, so max x = max_scope - (k_plus_1 - start_pos - 1)
        max_x = max_scope - (k_plus_1 - start_pos - 1)
        if max_x <= last_val:
            return None

        # Collect valid candidates
        valid = []
        for x in range(last_val + 1, max_x + 1):
            ok = True
            for a in row_so_far:
                d = x - a
                if (used_bits >> d) & 1:
                    ok = False
                    break
            if ok:
                valid.append(x)

        if not valid:
            return None

        # Limit candidates to keep search tractable
        # Prefer smaller values, but add some randomness
        n_small = min(8, len(valid))
        small_candidates = valid[:n_small]
        if len(valid) > n_small and rng.random() < 0.4:
            extra_pool = valid[n_small:]
            extra = rng.sample(extra_pool, min(4, len(extra_pool)))
            candidates = small_candidates + extra
        else:
            candidates = small_candidates

        rng.shuffle(candidates)

        for x in candidates:
            # Mark diffs
            mask = 0
            for a in row_so_far:
                d = x - a
                mask |= (1 << d)
            used_bits |= mask

            new_row = row_so_far + [x]
            res = build_row(start_pos + 1, x, new_row)
            if res is not None:
                return res

            used_bits &= ~mask

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        # Mark all diffs for this row
        for a in range(k_plus_1):
            for b in range(a + 1, k_plus_1):
                used_bits |= (1 << (row[b] - row[a]))

    # Verify
    all_diffs = set()
    for row in rows:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                d = row[j] - row[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)

    return rows
