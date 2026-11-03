import time
import random
import sys
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

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

    # 2. Local improvement
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Polish pass
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Matching-based compaction
    if time.time() < deadline - 10.0:
        matched = _matching_compact(best_solution, n_rows, k, deadline)
        if matched:
            m_scope = max(max(row) for row in matched)
            if m_scope < best_scope:
                best_scope = m_scope
                best_solution = matched
                improved = _improve_scope(best_solution, n_rows, k, deadline)
                i_scope = max(max(row) for row in improved)
                if i_scope < best_scope:
                    best_scope = i_scope
                    best_solution = improved

    # 5. Randomized backtracking for tighter scopes
    attempt = 0
    while time.time() < deadline - 8.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        for _ in range(40):
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

        if not found and target_scope <= lower_bound + 2:
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
    Backtracking with bitset tracking, MRV heuristic, and randomized candidate ordering.
    """
    rng = random.Random(seed_offset * 7919 + 12345)
    rows = []
    k_plus_1 = k + 1

    max_diff = max_scope
    used_bits = 0
    node_count = [0]

    def build_row(pos, current_mark, row_so_far, min_first_elem=1):
        nonlocal used_bits
        if time.time() > deadline:
            return None

        node_count[0] += 1
        if node_count[0] % 20000 == 0 and time.time() > deadline - 0.5:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        remaining = k_plus_1 - pos - 1
        max_x = max_scope - remaining
        if max_x <= current_mark:
            return None

        start_x = max(current_mark + 1, min_first_elem)
        if start_x > max_x:
            return None

        # Collect valid candidates
        valid_candidates = []
        for x in range(start_x, max_x + 1):
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

        if len(valid_candidates) < remaining + 1:
            return None

        # MRV-like: prefer candidates that leave more options for later positions
        # Approximate by trying smaller candidates first (they leave more room)
        n_smallest = min(25, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]

        # Occasionally explore larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra_count = min(8, len(valid_candidates) - n_smallest)
            extra = rng.sample(valid_candidates[n_smallest:], extra_count)
            candidates_to_try = candidates_to_try + extra

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            mask = 0
            for a in row_so_far:
                d = x - a
                mask |= (1 << d)
            old_bits = used_bits
            used_bits |= mask

            if pos == 1:
                next_min = x + 1
            else:
                next_min = 1

            res = build_row(pos + 1, x, row_so_far + [x], next_min)
            if res is not None:
                return res

            used_bits = old_bits

        return None

    for i in range(n_rows):
        if i == 0:
            min_first = 1
        else:
            min_first = rows[i - 1][1] + 1

        row = build_row(1, 0, [0], min_first)
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
    Scope reduction with pull-to-smallest, SA with single and 2-element moves.
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

    def _try_move(row_idx, pos, new_val):
        row = best[row_idx]
        old_val = row[pos]
        if new_val == old_val:
            return False
        if new_val <= row[pos - 1] or (pos < k_plus_1 - 1 and new_val >= row[pos + 1]):
            return False

        new_diffs = set()
        for i in range(pos):
            d = new_val - row[i]
            if d <= 0 or d in used_diffs:
                return False
            new_diffs.add(d)
        for j in range(pos + 1, k_plus_1):
            d = row[j] - new_val
            if d <= 0 or d in used_diffs:
                return False
            new_diffs.add(d)

        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for j in range(pos + 1, k_plus_1):
            old_diffs.add(row[j] - old_val)

        if new_diffs & (used_diffs - old_diffs):
            return False

        used_diffs.difference_update(old_diffs)
        used_diffs.update(new_diffs)
        row[pos] = new_val
        return True

    def _try_two_move(row_idx1, pos1, new_val1, row_idx2, pos2, new_val2):
        """Try moving two elements simultaneously (in different rows or same row)."""
        row1 = best[row_idx1]
        row2 = best[row_idx2]
        old_val1 = row1[pos1]
        old_val2 = row2[pos2]

        if new_val1 == old_val1 and new_val2 == old_val2:
            return False

        # Check bounds
        if new_val1 <= row1[pos1 - 1] or (pos1 < k_plus_1 - 1 and new_val1 >= row1[pos1 + 1]):
            return False
        if new_val2 <= row2[pos2 - 1] or (pos2 < k_plus_1 - 1 and new_val2 >= row2[pos2 + 1]):
            return False

        # Compute new differences for both moves
        new_diffs = set()
        for i in range(pos1):
            d = new_val1 - row1[i]
            if d <= 0:
                return False
            new_diffs.add(d)
        for j in range(pos1 + 1, k_plus_1):
            d = row1[j] - new_val1
            if d <= 0:
                return False
            new_diffs.add(d)

        for i in range(pos2):
            d = new_val2 - row2[i]
            if d <= 0:
                return False
            new_diffs.add(d)
        for j in range(pos2 + 1, k_plus_1):
            d = row2[j] - new_val2
            if d <= 0:
                return False
            new_diffs.add(d)

        # Compute old differences
        old_diffs = set()
        for i in range(pos1):
            old_diffs.add(old_val1 - row1[i])
        for j in range(pos1 + 1, k_plus_1):
            old_diffs.add(row1[j] - old_val1)

        for i in range(pos2):
            old_diffs.add(old_val2 - row2[i])
        for j in range(pos2 + 1, k_plus_1):
            old_diffs.add(row2[j] - old_val2)

        # Check for conflicts
        if new_diffs & (used_diffs - old_diffs):
            return False

        # Also check that new_diffs don't conflict with each other (within the same row)
        # This is already handled by the structure, but let's be safe
        if len(new_diffs) != len(list(new_diffs)):
            return False

        used_diffs.difference_update(old_diffs)
        used_diffs.update(new_diffs)
        row1[pos1] = new_val1
        row2[pos2] = new_val2
        return True

    # Phase 1: Pull to smallest
    for iteration in range(30):
        if time.time() > deadline - 8.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    # Phase 2: SA with single and 2-element moves
    current_solution = [list(row) for row in best]
    current_scope = max(max(row) for row in current_solution)
    best_sa = [list(row) for row in current_solution]
    best_sa_scope = current_scope

    used_diffs_sa = set()
    for row in current_solution:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs_sa.add(row[j] - row[i])

    def _try_move_sa(row_idx, pos, new_val):
        row = current_solution[row_idx]
        old_val = row[pos]
        if new_val == old_val:
            return False
        if new_val <= row[pos - 1] or (pos < k_plus_1 - 1 and new_val >= row[pos + 1]):
            return False
        new_diffs = set()
        for i in range(pos):
            d = new_val - row[i]
            if d <= 0 or d in used_diffs_sa:
                return False
            new_diffs.add(d)
        for j in range(pos + 1, k_plus_1):
            d = row[j] - new_val
            if d <= 0 or d in used_diffs_sa:
                return False
            new_diffs.add(d)
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for j in range(pos + 1, k_plus_1):
            old_diffs.add(row[j] - old_val)
        if new_diffs & (used_diffs_sa - old_diffs):
            return False
        used_diffs_sa.difference_update(old_diffs)
        used_diffs_sa.update(new_diffs)
        row[pos] = new_val
        return True

    def _undo_move_sa(row_idx, pos, old_val):
        row = current_solution[row_idx]
        new_val = row[pos]
        row[pos] = old_val
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, k_plus_1):
            new_diffs.add(row[j] - new_val)
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for j in range(pos + 1, k_plus_1):
            old_diffs.add(row[j] - old_val)
        used_diffs_sa.difference_update(new_diffs)
        used_diffs_sa.update(old_diffs)

    def _try_two_move_sa(row_idx1, pos1, new_val1, row_idx2, pos2, new_val2):
        row1 = current_solution[row_idx1]
        row2 = current_solution[row_idx2]
        old_val1 = row1[pos1]
        old_val2 = row2[pos2]
        if new_val1 == old_val1 and new_val2 == old_val2:
            return False
        if new_val1 <= row1[pos1 - 1] or (pos1 < k_plus_1 - 1 and new_val1 >= row1[pos1 + 1]):
            return False
        if new_val2 <= row2[pos2 - 1] or (pos2 < k_plus_1 - 1 and new_val2 >= row2[pos2 + 1]):
            return False
        new_diffs = set()
        for i in range(pos1):
            d = new_val1 - row1[i]
            if d <= 0:
                return False
            new_diffs.add(d)
        for j in range(pos1 + 1, k_plus_1):
            d = row1[j] - new_val1
            if d <= 0:
                return False
            new_diffs.add(d)
        for i in range(pos2):
            d = new_val2 - row2[i]
            if d <= 0:
                return False
            new_diffs.add(d)
        for j in range(pos2 + 1, k_plus_1):
            d = row2[j] - new_val2
            if d <= 0:
                return False
            new_diffs.add(d)
        old_diffs = set()
        for i in range(pos1):
            old_diffs.add(old_val1 - row1[i])
        for j in range(pos1 + 1, k_plus_1):
            old_diffs.add(row1[j] - old_val1)
        for i in range(pos2):
            old_diffs.add(old_val2 - row2[i])
        for j in range(pos2 + 1, k_plus_1):
            old_diffs.add(row2[j] - old_val2)
        if new_diffs & (used_diffs_sa - old_diffs):
            return False
        used_diffs_sa.difference_update(old_diffs)
        used_diffs_sa.update(new_diffs)
        row1[pos1] = new_val1
        row2[pos2] = new_val2
        return True

    def _undo_two_move_sa(row_idx1, pos1, old_val1, row_idx2, pos2, old_val2):
        row1 = current_solution[row_idx1]
        row2 = current_solution[row_idx2]
        new_val1 = row1[pos1]
        new_val2 = row2[pos2]
        row1[pos1] = old_val1
        row2[pos2] = old_val2
        new_diffs = set()
        for i in range(pos1):
            new_diffs.add(new_val1 - row1[i])
        for j in range(pos1 + 1, k_plus_1):
            new_diffs.add(row1[j] - new_val1)
        for i in range(pos2):
            new_diffs.add(new_val2 - row2[i])
        for j in range(pos2 + 1, k_plus_1):
            new_diffs.add(row2[j] - new_val2)
        old_diffs = set()
        for i in range(pos1):
            old_diffs.add(old_val1 - row1[i])
        for j in range(pos1 + 1, k_plus_1):
            old_diffs.add(row1[j] - old_val1)
        for i in range(pos2):
            old_diffs.add(old_val2 - row2[i])
        for j in range(pos2 + 1, k_plus_1):
            old_diffs.add(row2[j] - old_val2)
        used_diffs_sa.difference_update(new_diffs)
        used_diffs_sa.update(old_diffs)

    # SA passes with adaptive temperature
    for sa_pass in range(5):
        total_steps = 400
        start_temp = 8.0 if sa_pass == 0 else (5.0 if sa_pass == 1 else (3.0 if sa_pass == 2 else (1.5 if sa_pass == 3 else 0.8)))
        end_temp = 0.02
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, total_steps - 1))

        for step in range(total_steps):
            if time.time() > deadline - 8.0:
                break

            temp = start_temp * (cooling_rate ** step)

            # Choose move type: single (70%) or two-element (30%)
            if random.random() < 0.7:
                # Single element move
                row_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k_plus_1 - 1)
                row = current_solution[row_idx]
                old_val = row[pos]

                if temp > 3.0:
                    offset = random.choice([-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10, 12])
                elif temp > 1.0:
                    offset = random.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8])
                else:
                    offset = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5])
                new_val = old_val + offset

                if _try_move_sa(row_idx, pos, new_val):
                    new_scope = max(max(r) for r in current_solution)
                    delta = new_scope - current_scope
                    if delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
                        current_scope = new_scope
                        if new_scope < best_sa_scope:
                            best_sa_scope = new_scope
                            best_sa = [list(r) for r in current_solution]
                    else:
                        _undo_move_sa(row_idx, pos, old_val)
            else:
                # Two-element move
                if n_rows >= 2:
                    row_idx1 = random.randint(0, n_rows - 1)
                    row_idx2 = random.randint(0, n_rows - 1)
                    pos1 = random.randint(1, k_plus_1 - 1)
                    pos2 = random.randint(1, k_plus_1 - 1)

                    row1 = current_solution[row_idx1]
                    row2 = current_solution[row_idx2]
                    old_val1 = row1[pos1]
                    old_val2 = row2[pos2]

                    if temp > 3.0:
                        off1 = random.choice([-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8])
                        off2 = random.choice([-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8])
                    elif temp > 1.0:
                        off1 = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                        off2 = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                    else:
                        off1 = random.choice([-3, -2, -1, 1, 2, 3])
                        off2 = random.choice([-3, -2, -1, 1, 2, 3])

                    new_val1 = old_val1 + off1
                    new_val2 = old_val2 + off2

                    if _try_two_move_sa(row_idx1, pos1, new_val1, row_idx2, pos2, new_val2):
                        new_scope = max(max(r) for r in current_solution)
                        delta = new_scope - current_scope
                        if delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
                            current_scope = new_scope
                            if new_scope < best_sa_scope:
                                best_sa_scope = new_scope
                                best_sa = [list(r) for r in current_solution]
                        else:
                            _undo_two_move_sa(row_idx1, pos1, old_val1, row_idx2, pos2, old_val2)

    if best_sa_scope < best_scope:
        best = best_sa
        best_scope = best_sa_scope

    # Phase 3: Final pull
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    for iteration in range(25):
        if time.time() > deadline - 5.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    new_scope = max(max(row) for row in best)
    return best if new_scope < best_scope else [list(row) for row in solution]


def _polish_scope(solution, n_rows, k, deadline):
    """
    High-precision polish with multiple SA restarts and 2-element moves.
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

    def _sa_pass(current, current_scope, max_steps=350):
        sol = [list(r) for r in current]
        scope = current_scope

        used = set()
        for row in sol:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    used.add(row[j] - row[i])

        def try_move(row_idx, pos, new_val):
            row = sol[row_idx]
            old_val = row[pos]
            if new_val == old_val:
                return False
            if new_val <= row[pos - 1] or (pos < k_plus_1 - 1 and new_val >= row[pos + 1]):
                return False
            new_diffs = set()
            for i in range(pos):
                d = new_val - row[i]
                if d <= 0 or d in used:
                    return False
                new_diffs.add(d)
            for j in range(pos + 1, k_plus_1):
                d = row[j] - new_val
                if d <= 0 or d in used:
                    return False
                new_diffs.add(d)
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, k_plus_1):
                old_diffs.add(row[j] - old_val)
            if new_diffs & (used - old_diffs):
                return False
            used.difference_update(old_diffs)
            used.update(new_diffs)
            row[pos] = new_val
            return True

        def undo_move(row_idx, pos, old_val):
            row = sol[row_idx]
            new_val = row[pos]
            row[pos] = old_val
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for j in range(pos + 1, k_plus_1):
                new_diffs.add(row[j] - new_val)
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, k_plus_1):
                old_diffs.add(row[j] - old_val)
            used.difference_update(new_diffs)
            used.update(old_diffs)

        def try_two_move(row_idx1, pos1, new_val1, row_idx2, pos2, new_val2):
            row1 = sol[row_idx1]
            row2 = sol[row_idx2]
            old_val1 = row1[pos1]
            old_val2 = row2[pos2]
            if new_val1 == old_val1 and new_val2 == old_val2:
                return False
            if new_val1 <= row1[pos1 - 1] or (pos1 < k_plus_1 - 1 and new_val1 >= row1[pos1 + 1]):
                return False
            if new_val2 <= row2[pos2 - 1] or (pos2 < k_plus_1 - 1 and new_val2 >= row2[pos2 + 1]):
                return False
            new_diffs = set()
            for i in range(pos1):
                d = new_val1 - row1[i]
                if d <= 0:
                    return False
                new_diffs.add(d)
            for j in range(pos1 + 1, k_plus_1):
                d = row1[j] - new_val1
                if d <= 0:
                    return False
                new_diffs.add(d)
            for i in range(pos2):
                d = new_val2 - row2[i]
                if d <= 0:
                    return False
                new_diffs.add(d)
            for j in range(pos2 + 1, k_plus_1):
                d = row2[j] - new_val2
                if d <= 0:
                    return False
                new_diffs.add(d)
            old_diffs = set()
            for i in range(pos1):
                old_diffs.add(old_val1 - row1[i])
            for j in range(pos1 + 1, k_plus_1):
                old_diffs.add(row1[j] - old_val1)
            for i in range(pos2):
                old_diffs.add(old_val2 - row2[i])
            for j in range(pos2 + 1, k_plus_1):
                old_diffs.add(row2[j] - old_val2)
            if new_diffs & (used - old_diffs):
                return False
            used.difference_update(old_diffs)
            used.update(new_diffs)
            row1[pos1] = new_val1
            row2[pos2] = new_val2
            return True

        def undo_two_move(row_idx1, pos1, old_val1, row_idx2, pos2, old_val2):
            row1 = sol[row_idx1]
            row2 = sol[row_idx2]
            new_val1 = row1[pos1]
            new_val2 = row2[pos2]
            row1[pos1] = old_val1
            row2[pos2] = old_val2
            new_diffs = set()
            for i in range(pos1):
                new_diffs.add(new_val1 - row1[i])
            for j in range(pos1 + 1, k_plus_1):
                new_diffs.add(row1[j] - new_val1)
            for i in range(pos2):
                new_diffs.add(new_val2 - row2[i])
            for j in range(pos2 + 1, k_plus_1):
                new_diffs.add(row2[j] - new_val2)
            old_diffs = set()
            for i in range(pos1):
                old_diffs.add(old_val1 - row1[i])
            for j in range(pos1 + 1, k_plus_1):
                old_diffs.add(row1[j] - old_val1)
            for i in range(pos2):
                old_diffs.add(old_val2 - row2[i])
            for j in range(pos2 + 1, k_plus_1):
                old_diffs.add(row2[j] - old_val2)
            used.difference_update(new_diffs)
            used.update(old_diffs)

        start_temp = 6.0
        end_temp = 0.01
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_steps - 1))

        for step in range(max_steps):
            if time.time() > deadline - 4.0:
                break
            temp = start_temp * (cooling_rate ** step)

            if random.random() < 0.7:
                row_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k_plus_1 - 1)
                row = sol[row_idx]
                old_val = row[pos]
                if temp > 3.0:
                    offset = random.choice([-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8])
                elif temp > 1.0:
                    offset = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                else:
                    offset = random.choice([-3, -2, -1, 1, 2, 3])
                new_val = old_val + offset
                if try_move(row_idx, pos, new_val):
                    new_scope = max(max(r) for r in sol)
                    delta = new_scope - scope
                    if delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
                        scope = new_scope
                    else:
                        undo_move(row_idx, pos, old_val)
            else:
                if n_rows >= 2:
                    row_idx1 = random.randint(0, n_rows - 1)
                    row_idx2 = random.randint(0, n_rows - 1)
                    pos1 = random.randint(1, k_plus_1 - 1)
                    pos2 = random.randint(1, k_plus_1 - 1)
                    row1 = sol[row_idx1]
                    row2 = sol[row_idx2]
                    old_val1 = row1[pos1]
                    old_val2 = row2[pos2]
                    if temp > 3.0:
                        off1 = random.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7])
                        off2 = random.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7])
                    else:
                        off1 = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                        off2 = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                    new_val1 = old_val1 + off1
                    new_val2 = old_val2 + off2
                    if try_two_move(row_idx1, pos1, new_val1, row_idx2, pos2, new_val2):
                        new_scope = max(max(r) for r in sol)
                        delta = new_scope - scope
                        if delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
                            scope = new_scope
                        else:
                            undo_two_move(row_idx1, pos1, old_val1, row_idx2, pos2, old_val2)
        return sol, scope

    # Initial pull
    _pull_all()
    best_scope = max(max(row) for row in best)

    # Multiple SA restarts with perturbation
    for restart in range(5):
        if time.time() > deadline - 4.0:
            break
        perturbed = [list(row) for row in best]
        for _ in range(8):
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k_plus_1 - 1)
            offset = random.choice([2, 3, 5, 7, 10])
            perturbed[row_idx][pos] += offset
            _pull_all()

        sol, scope = _sa_pass(perturbed, max(max(row) for row in perturbed))
        if scope < best_scope:
            best = sol
            best_scope = scope
            used_diffs = set()
            for row in best:
                for i in range(k_plus_1):
                    for j in range(i + 1, k_plus_1):
                        used_diffs.add(row[j] - row[i])
            _pull_all()
            best_scope = max(max(row) for row in best)

    # Final pull
    _pull_all()
    new_scope = max(max(row) for row in best)
    return best if new_scope < best_scope else [list(row) for row in solution]


def _matching_compact(solution, n_rows, k, deadline):
    """
    Use bipartite matching on difference slots to drive compactness.
    For each row, try to reassign elements to minimize the maximum entry
    while maintaining distinctness of all differences.
    """
    if not solution:
        return None

    k_plus_1 = k + 1
    current_scope = max(max(row) for row in solution)

    # Collect all used differences
    all_diffs = set()
    for row in solution:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                all_diffs.add(row[j] - row[i])

    best = [list(row) for row in solution]

    # Try to reduce scope by matching elements to smaller positions
    # For each row, consider moving each element to a smaller value
    # and use matching to find a consistent assignment

    for attempt in range(3):
        if time.time() > deadline - 5.0:
            break

        # Build a cost matrix for each row: cost of placing element at position p
        # with value v is v^2 (to drive compactness)
        # But we need to ensure all differences remain distinct

        # Simpler approach: iteratively try to pull each element down
        # using a matching-like greedy with conflict checking

        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):  # Right to left
                # Try to reduce this element
                for new_val in range(row[pos - 1] + 1, row[pos]):
                    # Check if this creates a conflict
                    conflict = False
                    for i in range(pos):
                        d = new_val - row[i]
                        if d <= 0:
                            break
                        # Check if d is used by another row or by this row at a different pair
                        # We need to check against all_diffs minus the differences involving this element
                        # This is complex, so use a simpler check
                        if d in all_diffs:
                            # Check if d is only used by differences involving position pos
                            # For simplicity, skip if d is in all_diffs
                            # (This is conservative but safe)
                            conflict = True
                            break
                    if conflict:
                        continue

                    # Also check differences with elements after pos
                    ok = True
                    for j in range(pos + 1, k_plus_1):
                        d = row[j] - new_val
                        if d <= 0:
                            ok = False
                            break
                        if d in all_diffs:
                            ok = False
                            break
                    if ok:
                        # Update all_diffs
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(row[pos] - row[i])
                        for j in range(pos + 1, k_plus_1):
                            old_diffs.add(row[j] - row[pos])
                        all_diffs.difference_update(old_diffs)

                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for j in range(pos + 1, k_plus_1):
                            new_diffs.add(row[j] - new_val)
                        all_diffs.update(new_diffs)

                        row[pos] = new_val
                        improved = True
                        break

        if not improved:
            break

    new_scope = max(max(row) for row in best)
    if new_scope < current_scope:
        return best
    return None
