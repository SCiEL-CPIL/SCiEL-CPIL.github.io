import time
import random
import sys
import numpy as np
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # differences are 1..M, so max entry >= num_diffs - 1

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
    # for target scopes below current best
    attempt = 0
    while time.time() < deadline - 10.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        # Try multiple random restarts for this target scope
        found = False
        for _ in range(80):
            if time.time() > deadline - 10.0:
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
    Backtracking search with MRV heuristic, bitset tracking, and better pruning.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1

    max_diff = max_scope
    used_bits = 0  # Python int as arbitrary-precision bitset

    def build_row(pos, current_mark, row_so_far, depth):
        nonlocal used_bits
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        # Prune upper bound: need at least (k_plus_1 - pos - 1) more elements
        remaining = k_plus_1 - pos - 1
        max_x = max_scope - remaining
        if max_x <= current_mark:
            return None

        # Find valid candidates for current position
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
            if len(valid_candidates) >= 40:  # Limit for speed
                break

        if not valid_candidates:
            return None

        # Check if we can complete the row with these candidates
        # Simple feasibility check: enough room for remaining elements
        if len(valid_candidates) < remaining + 1:
            return None

        # Strategy: Try smallest candidates first (greedy), with some randomness
        n_smallest = min(15, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]

        # Occasionally try larger candidates to escape local minima
        if len(valid_candidates) > n_smallest and rng.random() < 0.4:
            extra_count = min(8, len(valid_candidates) - n_smallest)
            extra = rng.sample(valid_candidates[n_smallest:], extra_count)
            candidates_to_try = candidates_to_try + extra

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            # Mark new differences using bit operations
            mask = 0
            for a in row_so_far:
                d = x - a
                mask |= (1 << d)
            old_bits = used_bits
            used_bits |= mask

            res = build_row(pos + 1, x, row_so_far + [x], depth + 1)
            if res is not None:
                return res

            # Unmark differences
            used_bits = old_bits

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0], 0)
        if row is None:
            return None
        rows.append(row)
        # Mark all differences in this row as used
        mask = 0
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                mask |= (1 << d)
        used_bits |= mask

    return rows


def _improve_scope(solution, n_rows, k, deadline):
    """
    Aggressive scope reduction with 2-element swaps and simulated annealing.
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
        """Pull all elements in row row_idx to their smallest feasible values."""
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
        """Try moving element at (row_idx, pos) to new_val. Returns True if successful."""
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

    def _try_swap(row_idx, pos1, pos2):
        """Try swapping elements at positions pos1 and pos2 in row row_idx."""
        if pos1 >= pos2:
            return False
        row = best[row_idx]
        val1 = row[pos1]
        val2 = row[pos2]
        if val1 == val2:
            return False

        # After swap, we need row[pos1] = val2 and row[pos2] = val1
        # Check strict increase: row[pos1-1] < val2 < val2 < ... < row[pos2]
        # Actually: row[pos1-1] < val2 and val2 < row[pos1+1] (if pos1+1 != pos2)
        # and row[pos2-1] < val1 and val1 < row[pos2+1] (if pos2+1 != pos1)
        
        # Check if swap maintains strict increase
        if pos1 + 1 == pos2:
            # Adjacent: need row[pos1-1] < val2 and val2 < val1 and val1 < row[pos2+1]
            if pos1 > 0 and row[pos1-1] >= val2:
                return False
            if val2 >= val1:
                return False
            if pos2 < k_plus_1 - 1 and val1 >= row[pos2+1]:
                return False
        else:
            # Non-adjacent: need row[pos1-1] < val2 and val2 < row[pos1+1]
            # and row[pos2-1] < val1 and val1 < row[pos2+1]
            if pos1 > 0 and row[pos1-1] >= val2:
                return False
            if row[pos1+1] <= val2:
                return False
            if row[pos2-1] >= val1:
                return False
            if pos2 < k_plus_1 - 1 and row[pos2+1] <= val1:
                return False

        # Compute new differences
        new_diffs = set()
        # Differences involving pos1 (now val2)
        for i in range(pos1):
            d = val2 - row[i]
            if d <= 0 or d in used_diffs:
                return False
            new_diffs.add(d)
        for j in range(pos1 + 1, k_plus_1):
            if j == pos2:
                continue
            d = row[j] - val2
            if d <= 0 or d in used_diffs:
                return False
            new_diffs.add(d)
        # Difference between pos1 and pos2
        d12 = val1 - val2
        if d12 <= 0 or d12 in used_diffs:
            return False
        new_diffs.add(d12)
        
        # Differences involving pos2 (now val1)
        for i in range(pos2):
            if i == pos1:
                continue
            d = val1 - row[i]
            if d <= 0 or d in used_diffs:
                return False
            new_diffs.add(d)
        for j in range(pos2 + 1, k_plus_1):
            d = row[j] - val1
            if d <= 0 or d in used_diffs:
                return False
            new_diffs.add(d)

        # Compute old differences
        old_diffs = set()
        for i in range(pos1):
            old_diffs.add(val1 - row[i])
        for j in range(pos1 + 1, k_plus_1):
            if j == pos2:
                continue
            old_diffs.add(row[j] - val1)
        old_diffs.add(val2 - val1)  # This is negative, shouldn't be in set
        # Actually old diff between pos1 and pos2 is val2 - val1
        old_diffs.discard(val2 - val1)
        old_diffs.add(val2 - val1)
        
        for i in range(pos2):
            if i == pos1:
                continue
            old_diffs.add(val2 - row[i])
        for j in range(pos2 + 1, k_plus_1):
            old_diffs.add(row[j] - val2)

        if new_diffs & (used_diffs - old_diffs):
            return False

        used_diffs.difference_update(old_diffs)
        used_diffs.update(new_diffs)
        row[pos1] = val2
        row[pos2] = val1
        return True

    # Phase 1: Pull to smallest
    for iteration in range(30):
        if time.time() > deadline - 15.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    # Phase 2: Simulated annealing with swaps and moves
    current_solution = [list(row) for row in best]
    current_scope = max(max(row) for row in current_solution)
    best_sa = [list(row) for row in current_solution]
    best_sa_scope = current_scope

    # Rebuild used_diffs for current_solution
    used_diffs_sa = set()
    for row in current_solution:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs_sa.add(row[j] - row[i])

    def _try_move_sa(row_idx, pos, new_val):
        """Try move in SA context."""
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
        """Undo a move in SA context."""
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

    def _try_swap_sa(row_idx, pos1, pos2):
        """Try swap in SA context."""
        if pos1 >= pos2:
            return False
        row = current_solution[row_idx]
        val1 = row[pos1]
        val2 = row[pos2]
        if val1 == val2:
            return False

        if pos1 + 1 == pos2:
            if pos1 > 0 and row[pos1-1] >= val2:
                return False
            if val2 >= val1:
                return False
            if pos2 < k_plus_1 - 1 and val1 >= row[pos2+1]:
                return False
        else:
            if pos1 > 0 and row[pos1-1] >= val2:
                return False
            if row[pos1+1] <= val2:
                return False
            if row[pos2-1] >= val1:
                return False
            if pos2 < k_plus_1 - 1 and row[pos2+1] <= val1:
                return False

        new_diffs = set()
        for i in range(pos1):
            d = val2 - row[i]
            if d <= 0 or d in used_diffs_sa:
                return False
            new_diffs.add(d)
        for j in range(pos1 + 1, k_plus_1):
            if j == pos2:
                continue
            d = row[j] - val2
            if d <= 0 or d in used_diffs_sa:
                return False
            new_diffs.add(d)
        d12 = val1 - val2
        if d12 <= 0 or d12 in used_diffs_sa:
            return False
        new_diffs.add(d12)
        
        for i in range(pos2):
            if i == pos1:
                continue
            d = val1 - row[i]
            if d <= 0 or d in used_diffs_sa:
                return False
            new_diffs.add(d)
        for j in range(pos2 + 1, k_plus_1):
            d = row[j] - val1
            if d <= 0 or d in used_diffs_sa:
                return False
            new_diffs.add(d)

        old_diffs = set()
        for i in range(pos1):
            old_diffs.add(val1 - row[i])
        for j in range(pos1 + 1, k_plus_1):
            if j == pos2:
                continue
            old_diffs.add(row[j] - val1)
        old_diffs.add(val2 - val1)
        
        for i in range(pos2):
            if i == pos1:
                continue
            old_diffs.add(val2 - row[i])
        for j in range(pos2 + 1, k_plus_1):
            old_diffs.add(row[j] - val2)

        if new_diffs & (used_diffs_sa - old_diffs):
            return False

        used_diffs_sa.difference_update(old_diffs)
        used_diffs_sa.update(new_diffs)
        row[pos1] = val2
        row[pos2] = val1
        return True

    def _undo_swap_sa(row_idx, pos1, pos2):
        """Undo a swap in SA context."""
        row = current_solution[row_idx]
        val1 = row[pos1]
        val2 = row[pos2]
        row[pos1] = val2
        row[pos2] = val1

        new_diffs = set()
        for i in range(pos1):
            new_diffs.add(val1 - row[i])
        for j in range(pos1 + 1, k_plus_1):
            if j == pos2:
                continue
            new_diffs.add(row[j] - val1)
        new_diffs.add(val2 - val1)
        
        for i in range(pos2):
            if i == pos1:
                continue
            new_diffs.add(val2 - row[i])
        for j in range(pos2 + 1, k_plus_1):
            new_diffs.add(row[j] - val2)

        old_diffs = set()
        for i in range(pos1):
            old_diffs.add(val2 - row[i])
        for j in range(pos1 + 1, k_plus_1):
            if j == pos2:
                continue
            old_diffs.add(row[j] - val2)
        old_diffs.add(val1 - val2)
        
        for i in range(pos2):
            if i == pos1:
                continue
            old_diffs.add(val1 - row[i])
        for j in range(pos2 + 1, k_plus_1):
            old_diffs.add(row[j] - val1)

        used_diffs_sa.difference_update(new_diffs)
        used_diffs_sa.update(old_diffs)

    total_steps = 800
    start_temp = 12.0
    end_temp = 0.05
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, total_steps - 1))

    for step in range(total_steps):
        if time.time() > deadline - 15.0:
            break

        temp = start_temp * (cooling_rate ** step)
        row_idx = random.randint(0, n_rows - 1)

        # Choose between move and swap
        if random.random() < 0.5:
            # Single element move
            pos = random.randint(1, k_plus_1 - 1)
            row = current_solution[row_idx]
            old_val = row[pos]
            offset = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 8, 12])
            new_val = old_val + offset

            if _try_move_sa(row_idx, pos, new_val):
                new_scope = max(max(r) for r in current_solution)
                delta = new_scope - current_scope

                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    current_scope = new_scope
                    if new_scope < best_sa_scope:
                        best_sa_scope = new_scope
                        best_sa = [list(r) for r in current_solution]
                else:
                    _undo_move_sa(row_idx, pos, old_val)
        else:
            # Two element swap
            pos1 = random.randint(1, k_plus_1 - 2)
            pos2 = random.randint(pos1 + 1, k_plus_1 - 1)

            if _try_swap_sa(row_idx, pos1, pos2):
                new_scope = max(max(r) for r in current_solution)
                delta = new_scope - current_scope

                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    current_scope = new_scope
                    if new_scope < best_sa_scope:
                        best_sa_scope = new_scope
                        best_sa = [list(r) for r in current_solution]
                else:
                    _undo_swap_sa(row_idx, pos1, pos2)

    # Update best if SA found improvement
    if best_sa_scope < best_scope:
        best = best_sa
        best_scope = best_sa_scope

    # Phase 3: Final pull pass
    # Rebuild used_diffs for best
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    for iteration in range(25):
        if time.time() > deadline - 10.0:
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
    High-precision polish with multiple SA restarts and final greedy pull.
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

    def _pull_all():
        """Pull all rows to smallest feasible values."""
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

    def _sa_pass(current, current_scope, max_steps=500):
        """Run a simulated annealing pass with swaps and moves."""
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

        def try_swap(row_idx, pos1, pos2):
            if pos1 >= pos2:
                return False
            row = sol[row_idx]
            val1 = row[pos1]
            val2 = row[pos2]
            if val1 == val2:
                return False

            if pos1 + 1 == pos2:
                if pos1 > 0 and row[pos1-1] >= val2:
                    return False
                if val2 >= val1:
                    return False
                if pos2 < k_plus_1 - 1 and val1 >= row[pos2+1]:
                    return False
            else:
                if pos1 > 0 and row[pos1-1] >= val2:
                    return False
                if row[pos1+1] <= val2:
                    return False
                if row[pos2-1] >= val1:
                    return False
                if pos2 < k_plus_1 - 1 and row[pos2+1] <= val1:
                    return False

            new_diffs = set()
            for i in range(pos1):
                d = val2 - row[i]
                if d <= 0 or d in used:
                    return False
                new_diffs.add(d)
            for j in range(pos1 + 1, k_plus_1):
                if j == pos2:
                    continue
                d = row[j] - val2
                if d <= 0 or d in used:
                    return False
                new_diffs.add(d)
            d12 = val1 - val2
            if d12 <= 0 or d12 in used:
                return False
            new_diffs.add(d12)
            
            for i in range(pos2):
                if i == pos1:
                    continue
                d = val1 - row[i]
                if d <= 0 or d in used:
                    return False
                new_diffs.add(d)
            for j in range(pos2 + 1, k_plus_1):
                d = row[j] - val1
                if d <= 0 or d in used:
                    return False
                new_diffs.add(d)

            old_diffs = set()
            for i in range(pos1):
                old_diffs.add(val1 - row[i])
            for j in range(pos1 + 1, k_plus_1):
                if j == pos2:
                    continue
                old_diffs.add(row[j] - val1)
            old_diffs.add(val2 - val1)
            
            for i in range(pos2):
                if i == pos1:
                    continue
                old_diffs.add(val2 - row[i])
            for j in range(pos2 + 1, k_plus_1):
                old_diffs.add(row[j] - val2)

            if new_diffs & (used - old_diffs):
                return False

            used.difference_update(old_diffs)
            used.update(new_diffs)
            row[pos1] = val2
            row[pos2] = val1
            return True

        def undo_swap(row_idx, pos1, pos2):
            row = sol[row_idx]
            val1 = row[pos1]
            val2 = row[pos2]
            row[pos1] = val2
            row[pos2] = val1

            new_diffs = set()
            for i in range(pos1):
                new_diffs.add(val1 - row[i])
            for j in range(pos1 + 1, k_plus_1):
                if j == pos2:
                    continue
                new_diffs.add(row[j] - val1)
            new_diffs.add(val2 - val1)
            
            for i in range(pos2):
                if i == pos1:
                    continue
                new_diffs.add(val2 - row[i])
            for j in range(pos2 + 1, k_plus_1):
                new_diffs.add(row[j] - val2)

            old_diffs = set()
            for i in range(pos1):
                old_diffs.add(val2 - row[i])
            for j in range(pos1 + 1, k_plus_1):
                if j == pos2:
                    continue
                old_diffs.add(row[j] - val2)
            old_diffs.add(val1 - val2)
            
            for i in range(pos2):
                if i == pos1:
                    continue
                old_diffs.add(val1 - row[i])
            for j in range(pos2 + 1, k_plus_1):
                old_diffs.add(row[j] - val1)

            used.difference_update(new_diffs)
            used.update(old_diffs)

        start_temp = 10.0
        end_temp = 0.03
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_steps - 1))

        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            temp = start_temp * (cooling_rate ** step)
            row_idx = random.randint(0, n_rows - 1)

            if random.random() < 0.5:
                # Single element move
                pos = random.randint(1, k_plus_1 - 1)
                row = sol[row_idx]
                old_val = row[pos]
                offset = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 8, 12])
                new_val = old_val + offset

                if try_move(row_idx, pos, new_val):
                    new_scope = max(max(r) for r in sol)
                    delta = new_scope - scope
                    if delta <= 0 or random.random() < math.exp(-delta / temp):
                        scope = new_scope
                    else:
                        undo_move(row_idx, pos, old_val)
            else:
                # Two element swap
                pos1 = random.randint(1, k_plus_1 - 2)
                pos2 = random.randint(pos1 + 1, k_plus_1 - 1)

                if try_swap(row_idx, pos1, pos2):
                    new_scope = max(max(r) for r in sol)
                    delta = new_scope - scope
                    if delta <= 0 or random.random() < math.exp(-delta / temp):
                        scope = new_scope
                    else:
                        undo_swap(row_idx, pos1, pos2)
        return sol, scope

    # Initial pull
    _pull_all()
    best_scope = max(max(row) for row in best)

    # Multiple SA restarts
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break
        # Perturb current best
        perturbed = [list(row) for row in best]
        for _ in range(8):
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k_plus_1 - 1)
            offset = random.choice([2, 3, 5, 8, 12])
            perturbed[row_idx][pos] += offset
            # Re-pull to fix
            _pull_all()

        sol, scope = _sa_pass(perturbed, max(max(row) for row in perturbed))
        if scope < best_scope:
            best = sol
            best_scope = scope
            # Rebuild used_diffs
            used_diffs = set()
            for row in best:
                for i in range(k_plus_1):
                    for j in range(i + 1, k_plus_1):
                        used_diffs.add(row[j] - row[i])
            _pull_all()
            best_scope = max(max(row) for row in best)

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
