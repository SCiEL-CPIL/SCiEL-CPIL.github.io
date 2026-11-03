import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
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
    while time.time() < deadline - 8.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        for _ in range(30):
            if time.time() > deadline - 8.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    polished = _polish_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in polished)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = polished
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 3:
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
    Backtracking search with bitset-based difference tracking and aggressive pruning.
    Uses Python's arbitrary-precision integers as bitsets for fast conflict checks.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1

    used_bits = 0  # Python int as arbitrary-precision bitset

    def build_row(pos, current_mark, row_so_far, depth):
        nonlocal used_bits
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        # Prune upper bound: need at least (k_plus_1 - pos - 1) more values
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        # Find valid candidates
        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if (used_bits >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
            if len(valid_candidates) >= 60:
                break

        if not valid_candidates:
            return None

        # Bias towards smaller values, with some randomness
        n_smallest = min(12, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]

        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(6, len(valid_candidates) - n_smallest))
            candidates_to_try = candidates_to_try + extra

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            mask = 0
            for a in row_so_far:
                d = x - a
                mask |= (1 << d)
            old_bits = used_bits
            used_bits |= mask

            res = build_row(pos + 1, x, row_so_far + [x], depth + 1)
            if res is not None:
                return res

            used_bits = old_bits

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0], 0)
        if row is None:
            return None
        rows.append(row)
        mask = 0
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                mask |= (1 << d)
        used_bits |= mask

    return rows


def _improve_scope(solution, n_rows, k, deadline):
    """
    Aggressive scope reduction: repeatedly pull each element to its smallest
    feasible value, then try multi-element perturbations to escape local minima.
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
        return any_changed

    # Phase 1: Repeatedly pull all rows to smallest feasible
    for iteration in range(40):
        if time.time() > deadline - 5.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    # Phase 2: Multi-element perturbations to escape local minima
    rng = random.Random(42)
    for iteration in range(25):
        if time.time() > deadline - 5.0:
            break
        any_changed = False
        # Perturb 1-2 random elements upward, then pull back
        n_perturb = rng.randint(1, 2)
        for _ in range(n_perturb):
            row_idx = rng.randint(0, n_rows - 1)
            row = best[row_idx]
            pos = rng.randint(1, k_plus_1 - 1)
            original = row[pos]
            offset = rng.randint(2, 10)
            new_val = original + offset
            if new_val > 250:
                continue
            new_diffs = set()
            ok = True
            for i in range(pos):
                d = new_val - row[i]
                if d in used_diffs:
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
                    old_diffs.add(original - row[i])
                for j in range(pos + 1, k_plus_1):
                    old_diffs.add(row[j] - original)
                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                row[pos] = new_val
                any_changed = True

        # Pull back after perturbation
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True

        if not any_changed:
            break

    # Phase 3: Final pull pass
    for iteration in range(15):
        if time.time() > deadline - 5.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]


def _polish_scope(solution, n_rows, k, deadline):
    """
    High-precision polish: alternating pull-to-smallest with targeted
    perturbations and 2-element swap moves for finer compaction.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def _pull_all():
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
        return any_changed

    def _try_swap(row_idx, pos1, pos2):
        row = best[row_idx]
        v1, v2 = row[pos1], row[pos2]
        if v1 == v2 or v1 >= v2:
            return False
        if pos1 > 0 and v2 <= row[pos1 - 1]:
            return False
        if pos2 < k_plus_1 - 1 and v1 >= row[pos2 + 1]:
            return False
        # Check feasibility of swap
        new_diffs = set()
        ok = True
        for i in range(pos1):
            d = v2 - row[i]
            if d <= 0:
                ok = False
                break
            new_diffs.add(d)
        if ok:
            for i in range(pos1 + 1, pos2):
                d = v2 - row[i]
                if d <= 0 or d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
        if ok:
            for i in range(pos2 + 1, k_plus_1):
                d = row[i] - v2
                if d <= 0 or d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
        if ok:
            for i in range(pos1 + 1, pos2):
                d = v1 - row[i]
                if d <= 0 or d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
        if ok:
            for i in range(pos2 + 1, k_plus_1):
                d = row[i] - v1
                if d <= 0 or d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
        if not ok:
            return False
        # Remove old diffs
        old_diffs = set()
        for i in range(pos1):
            old_diffs.add(v1 - row[i])
        for i in range(pos1 + 1, pos2):
            old_diffs.add(v1 - row[i])
        for i in range(pos2 + 1, k_plus_1):
            old_diffs.add(row[i] - v1)
        for i in range(pos1):
            old_diffs.add(v2 - row[i])
        for i in range(pos1 + 1, pos2):
            old_diffs.add(v2 - row[i])
        for i in range(pos2 + 1, k_plus_1):
            old_diffs.add(row[i] - v2)
        old_diffs.add(v2 - v1)
        if not (new_diffs & (used_diffs - old_diffs)):
            used_diffs.difference_update(old_diffs)
            used_diffs.update(new_diffs)
            row[pos1], row[pos2] = v2, v1
            return True
        return False

    # Multiple rounds
    rng = random.Random(99)
    for round_num in range(50):
        if time.time() > deadline - 5.0:
            break

        _pull_all()

        # Every 3rd round, try a swap
        if round_num % 3 == 2:
            row_idx = round_num % n_rows
            row = best[row_idx]
            for pos1 in range(1, k_plus_1):
                for pos2 in range(pos1 + 1, k_plus_1):
                    if _try_swap(row_idx, pos1, pos2):
                        break
                else:
                    continue
                break

        # Perturbation every other round
        if round_num % 2 == 1:
            row_idx = rng.randint(0, n_rows - 1)
            row = best[row_idx]
            pos = rng.randint(1, k_plus_1 - 1)
            original = row[pos]
            for offset in [1, 2, 3, 5]:
                new_val = original + offset
                if new_val > 200:
                    break
                new_diffs = set()
                ok = True
                for i in range(pos):
                    d = new_val - row[i]
                    if d in used_diffs:
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
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    break

        _pull_all()

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
