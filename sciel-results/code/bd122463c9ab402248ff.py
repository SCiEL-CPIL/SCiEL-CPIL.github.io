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

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Local compaction (fast)
    best_solution = _local_compact(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Simulated annealing with numpy bitset (structural change)
    best_solution = _simulated_annealing(n_rows, k, best_scope, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Targeted scope reduction via backtracking
    while time.time() < deadline - 10.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        for attempt in range(30):
            if time.time() > deadline - 10.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Polish with local compaction
                    improved = _local_compact(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
                    break

        if not found:
            break

    # 5. Final polish
    best_solution = _local_compact(best_solution, n_rows, k, deadline)

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

def _local_compact(solution, n_rows, k, deadline):
    """Fast local search: pull each element to its smallest feasible position."""
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Build used differences set
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def _pull_row(row_idx):
        row = best[row_idx]
        any_changed = False
        for pos in range(1, k_plus_1):
            for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                # Check if new_val works
                conflict = False
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0 or d in used_diffs:
                        conflict = True
                        break
                if conflict:
                    continue
                for j in range(pos + 1, k_plus_1):
                    d = row[j] - new_val
                    if d <= 0 or d in used_diffs:
                        conflict = True
                        break
                if conflict:
                    continue

                # Remove old diffs
                for i in range(pos):
                    used_diffs.discard(row[pos] - row[i])
                for j in range(pos + 1, k_plus_1):
                    used_diffs.discard(row[j] - row[pos])

                # Add new diffs
                for i in range(pos):
                    used_diffs.add(new_val - row[i])
                for j in range(pos + 1, k_plus_1):
                    used_diffs.add(row[j] - new_val)

                row[pos] = new_val
                any_changed = True
                break
        return any_changed

    # Iterate until no more improvements
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for row_idx in range(n_rows):
            if _pull_row(row_idx):
                improved = True

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    return [list(row) for row in solution]

def _simulated_annealing(n_rows, k, initial_scope, deadline):
    """
    Simulated annealing using numpy bitset for fast conflict detection.
    This is a structural change from the previous local-search-only approach.
    """
    if time.time() > deadline - 5.0:
        return None

    k_plus_1 = k + 1
    max_diff = initial_scope + 50  # allow some headroom
    rng = random.Random(12345)

    # Build initial solution from greedy
    init_rows = greedy_construct(n_rows, k)
    if not init_rows:
        return None

    # Create numpy bitset for used differences
    used_bits = np.zeros(max_diff + 1, dtype=np.int32)

    def compute_diffs(row):
        diffs = []
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                diffs.append(row[j] - row[i])
        return diffs

    # Mark initial used differences
    for row in init_rows:
        for d in compute_diffs(row):
            if d <= max_diff:
                used_bits[d] = 1

    current_rows = [list(row) for row in init_rows]
    current_scope = max(max(row) for row in current_rows)

    best_rows = [list(row) for row in current_rows]
    best_scope = current_scope

    # Annealing parameters
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.9995
    T = T_start

    step = 0
    while time.time() < deadline - 5.0 and T > T_end:
        step += 1

        # Pick a random row and position to perturb
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k_plus_1 - 1)

        row = current_rows[row_idx]
        old_val = row[pos]

        # Try to move to a nearby value
        min_val = row[pos - 1] + 1
        max_val = row[pos + 1] - 1 if pos < k_plus_1 - 1 else min_val + 50

        if max_val <= min_val:
            T *= cooling_rate
            continue

        # Choose a new value
        if rng.random() < 0.5:
            new_val = rng.randint(min_val, max_val)
        else:
            # Small perturbation
            delta = rng.randint(-3, 3)
            new_val = old_val + delta
            if new_val < min_val or new_val > max_val:
                new_val = rng.randint(min_val, max_val)

        if new_val == old_val:
            T *= cooling_rate
            continue

        # Compute old and new differences for this position
        old_diffs = []
        new_diffs = []
        for i in range(k_plus_1):
            if i == pos:
                continue
            if i < pos:
                old_diffs.append(old_val - row[i])
                new_diffs.append(new_val - row[i])
            else:
                old_diffs.append(row[i] - old_val)
                new_diffs.append(row[i] - new_val)

        # Check for conflicts
        # Remove old diffs temporarily
        for d in old_diffs:
            if d <= max_diff:
                used_bits[d] = 0

        # Check if new diffs conflict
        conflict = False
        for d in new_diffs:
            if d <= 0:
                conflict = True
                break
            if d <= max_diff and used_bits[d] > 0:
                conflict = True
                break

        if conflict:
            # Restore old diffs
            for d in old_diffs:
                if d <= max_diff:
                    used_bits[d] = 1
            T *= cooling_rate
            continue

        # Accept the move (always accept for simplicity, or use SA criterion)
        # Compute new scope
        new_row = list(row)
        new_row[pos] = new_val
        new_scope = max(new_row[-1], max(r[-1] for r in current_rows if r is not new_row))

        # SA acceptance: accept if better, or with probability exp(-delta/T)
        delta = new_scope - current_scope
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            # Apply move
            row[pos] = new_val
            for d in new_diffs:
                if d <= max_diff:
                    used_bits[d] = 1
            current_scope = new_scope

            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Restore old diffs
            for d in old_diffs:
                if d <= max_diff:
                    used_bits[d] = 1

        T *= cooling_rate

    # Final compaction
    result = _local_compact(best_rows, n_rows, k, deadline)
    final_scope = max(max(row) for row in result)
    if final_scope < best_scope:
        return result
    return best_rows

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search with bitset-based difference tracking.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    k_plus_1 = k + 1
    max_diff = max_scope

    # Use numpy for fast bitset operations
    used_bits = np.zeros(max_diff + 1, dtype=np.int32)

    def build_row(pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        # Prune: need enough room for remaining elements
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        # Find valid candidates
        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if d <= max_diff and used_bits[d] > 0:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
            if len(valid_candidates) >= 30:
                break

        if not valid_candidates:
            return None

        # Bias towards smaller values
        n_smallest = min(12, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]

        # Occasionally add larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(6, len(valid_candidates) - n_smallest))
            candidates_to_try = candidates_to_try + extra

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            # Mark new differences
            new_diffs = []
            for a in row_so_far:
                d = x - a
                if d <= max_diff:
                    new_diffs.append(d)
                    used_bits[d] = 1

            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res

            # Unmark differences
            for d in new_diffs:
                used_bits[d] = 0

        return None

    rows = []
    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        # Mark all differences in this row
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                if d <= max_diff:
                    used_bits[d] = 1

    return rows
