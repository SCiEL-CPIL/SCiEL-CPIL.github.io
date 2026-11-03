import time
import random
import sys
import numpy as np


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1
    k_plus_1 = k + 1

    # 1. Fast greedy baseline
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. LNS: iteratively reconstruct rows to free up difference slots
    best_solution, best_scope = _lns_optimize(
        best_solution, n_rows, k, lower_bound, deadline
    )

    # 3. Final pull-to-smallest polish
    best_solution = _pull_polish(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Backtracking search for target scopes below current best
    attempt = 0
    while time.time() < deadline - 8.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        for _ in range(40):
            if time.time() > deadline - 8.0:
                break
            sol = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if sol:
                ns = max(max(row) for row in sol)
                if ns < best_scope:
                    best_scope = ns
                    best_solution = sol
                    found = True
                    # Polish the new solution
                    best_solution = _pull_polish(best_solution, n_rows, k, deadline)
                    best_scope = max(max(row) for row in best_solution)
                    # LNS on the improved solution
                    best_solution, best_scope = _lns_optimize(
                        best_solution, n_rows, k, lower_bound, deadline
                    )
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 3:
            break

    # Final polish
    best_solution = _pull_polish(best_solution, n_rows, k, deadline)
    return best_solution


def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _compute_used_diffs(solution, k_plus_1):
    used = set()
    for row in solution:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used.add(row[j] - row[i])
    return used


def _pull_polish(solution, n_rows, k, deadline):
    """Pull each element to its smallest feasible value, iteratively."""
    if not solution:
        return solution
    best = [list(row) for row in solution]
    k_plus_1 = k + 1
    used_diffs = _compute_used_diffs(best, k_plus_1)

    for iteration in range(40):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(1, k_plus_1):
                for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                    new_diffs = set()
                    ok = True
                    for i in range(pos):
                        d = new_val - row[i]
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if ok:
                        for j in range(pos + 1, k_plus_1):
                            d = row[j] - new_val
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                    if ok:
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(row[pos] - row[i])
                        for j in range(pos + 1, k_plus_1):
                            old_diffs.add(row[j] - row[pos])
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        row[pos] = new_val
                        any_changed = True
                        break
        if not any_changed:
            break

    new_scope = max(max(row) for row in best)
    orig_scope = max(max(row) for row in solution)
    if new_scope <= orig_scope:
        return best
    return [list(row) for row in solution]


def _lns_optimize(solution, n_rows, k, lower_bound, deadline):
    """
    Large Neighborhood Search: iteratively reconstruct one or more rows
    from scratch using the current used-diff set, then pull-to-smallest.
    This allows global restructuring that single-element moves cannot achieve.
    """
    if not solution:
        return solution, max(max(row) for row in solution) if solution else 0

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    rng = random.Random(42)

    for iteration in range(200):
        if time.time() > deadline - 8.0:
            break
        if best_scope <= lower_bound:
            break

        used_diffs = _compute_used_diffs(best, k_plus_1)

        # Choose which row(s) to reconstruct
        # Strategy: reconstruct the row with the largest last element first
        row_idx = max(range(n_rows), key=lambda i: best[i][-1])

        # Save the row's old differences
        old_row = best[row_idx]
        old_diffs = set()
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                old_diffs.add(old_row[j] - old_row[i])

        # Remove old row's differences from used set
        freed_diffs = used_diffs - old_diffs  # differences used by other rows
        # Actually we need: used_diffs without old_row's diffs
        # Let's recompute: after removing old_row, what's available?
        # used_diffs currently contains ALL diffs. Remove old_row's to get "other" diffs.
        other_diffs = set()
        for ri in range(n_rows):
            if ri == row_idx:
                continue
            row = best[ri]
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    other_diffs.add(row[j] - row[i])

        # Reconstruct the row
        new_row = _reconstruct_row(k_plus_1, other_diffs, deadline, rng, best_scope)

        if new_row is not None:
            new_scope_row = new_row[-1]
            # Compute new overall scope
            other_max = max(best[ri][-1] for ri in range(n_rows) if ri != row_idx)
            new_total_scope = max(new_scope_row, other_max)

            if new_total_scope < best_scope:
                best[row_idx] = new_row
                best_scope = new_total_scope

                # Pull-to-smallest pass on all rows
                used_diffs = _compute_used_diffs(best, k_plus_1)
                for _ in range(15):
                    if time.time() > deadline - 8.0:
                        break
                    any_changed = False
                    for ri in range(n_rows):
                        row = best[ri]
                        for pos in range(1, k_plus_1):
                            for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                                new_diffs = set()
                                ok = True
                                for i in range(pos):
                                    d = new_val - row[i]
                                    if d <= 0 or d in used_diffs:
                                        ok = False
                                        break
                                    new_diffs.add(d)
                                if ok:
                                    for j in range(pos + 1, k_plus_1):
                                        d = row[j] - new_val
                                        if d <= 0 or d in used_diffs:
                                            ok = False
                                            break
                                        new_diffs.add(d)
                                if ok:
                                    old_diffs_local = set()
                                    for i in range(pos):
                                        old_diffs_local.add(row[pos] - row[i])
                                    for j in range(pos + 1, k_plus_1):
                                        old_diffs_local.add(row[j] - row[pos])
                                    used_diffs.difference_update(old_diffs_local)
                                    used_diffs.update(new_diffs)
                                    row[pos] = new_val
                                    any_changed = True
                                    break
                    if not any_changed:
                        break

                best_scope = max(max(row) for row in best)
                if best_scope <= lower_bound:
                    break
        else:
            # Try a different row
            if iteration % 5 == 4:
                # Shuffle row order for next iterations
                pass

    return best, best_scope


def _reconstruct_row(k_plus_1, forbidden_diffs, deadline, rng, scope_hint):
    """
    Reconstruct a single row of k+1 increasing integers starting at 0,
    avoiding all differences in forbidden_diffs.
    Uses randomized greedy with backtracking.
    """
    if time.time() > deadline:
        return None

    # Try multiple random seeds
    for attempt in range(20):
        if time.time() > deadline:
            return None

        row = [0]
        x = 0
        success = True
        for pos in range(1, k_plus_1):
            # Find candidates
            candidates = []
            x = row[-1]
            max_x = scope_hint + 20  # allow slightly above current scope
            while x < max_x:
                x += 1
                ok = True
                for a in row:
                    d = x - a
                    if d in forbidden_diffs:
                        ok = False
                        break
                if ok:
                    candidates.append(x)
                if len(candidates) >= 30:
                    break

            if not candidates:
                success = False
                break

            # Pick from candidates with bias towards smaller
            if attempt == 0:
                # Greedy: pick smallest
                chosen = candidates[0]
            elif attempt % 2 == 0:
                # Pick from smallest 5
                n = min(5, len(candidates))
                chosen = rng.choice(candidates[:n])
            else:
                # Pick from smallest 15
                n = min(15, len(candidates))
                chosen = rng.choice(candidates[:n])

            row.append(chosen)

        if success and len(row) == k_plus_1:
            return row

    return None


def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search for a solution with max entry <= max_scope.
    """
    rng = random.Random(seed_offset * 7919 + 31337)
    k_plus_1 = k + 1

    used_diffs = [False] * (max_scope + 1)
    rows = []

    def build_row(pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        valid = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid.append(x)
            if len(valid) >= 40:
                break

        if not valid:
            return None

        n_small = min(12, len(valid))
        to_try = valid[:n_small]
        if len(valid) > n_small and rng.random() < 0.3:
            extra = rng.sample(valid[n_small:], min(6, len(valid) - n_small))
            to_try = to_try + extra
        rng.shuffle(to_try)

        for x in to_try:
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)
                used_diffs[d] = True

            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res

            for d in new_diffs:
                used_diffs[d] = False

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                if not used_diffs[d]:
                    used_diffs[d] = True

    return rows
