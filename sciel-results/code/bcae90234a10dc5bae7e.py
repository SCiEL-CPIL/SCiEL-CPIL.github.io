import time
import random
import math
import sys


def _row_diffs_list(row):
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _row_mask(row):
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))


def _greedy_construct(n_rows, num_elements, rng, scope_hint):
    """
    Greedy column-wise construction with a given scope hint.
    Returns a valid set of rows or None if it can't complete within scope_hint.
    """
    rows = [[0] for _ in range(n_rows)]
    used_mask = 0

    for col in range(1, num_elements):
        order = list(range(n_rows))
        rng.shuffle(order)

        for r in order:
            row = rows[r]
            last_val = row[-1]
            found = False

            for val in range(last_val + 1, scope_hint + 1):
                new_diffs = []
                valid = True
                for prev in row:
                    d = val - prev
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if not valid:
                    continue

                conflict = False
                for d in new_diffs:
                    if (used_mask >> d) & 1:
                        conflict = True
                        break

                if conflict:
                    continue

                row.append(val)
                for d in new_diffs:
                    used_mask |= (1 << d)
                found = True
                break

            if not found:
                return None

    return rows


def _sa_energy(rows, num_elements):
    """
    Energy function: 0 if valid, otherwise count of duplicate differences.
    Lower is better. We want energy = 0 and minimum scope.
    """
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    n = len(all_diffs)
    unique = len(set(all_diffs))
    return n - unique  # number of duplicate differences


def _sa_scope_penalty(rows, num_elements, counting_bound):
    """Penalty for exceeding counting bound."""
    s = _scope(rows)
    if s <= counting_bound:
        return 0
    return (s - counting_bound) * 10


def _sa_total_cost(rows, num_elements, counting_bound):
    """Total cost = energy + scope penalty."""
    return _sa_energy(rows, num_elements) + _sa_scope_penalty(rows, num_elements, counting_bound)


def _simulated_annealing(initial_rows, n_rows, num_elements, counting_bound, rng, deadline):
    """
    Simulated annealing to minimize the total cost.
    Moves: pick a random row and position, try a nearby value.
    """
    start_time = time.time()
    current = [r[:] for r in initial_rows]
    current_cost = _sa_total_cost(current, num_elements, counting_bound)
    best = [r[:] for r in current]
    best_cost = current_cost

    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    n_steps = 500000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))

    T = T_start

    for step in range(n_steps):
        if time.time() > deadline:
            break

        # Pick a random row and position to perturb
        r = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, num_elements - 1)
        row = current[r]
        old_val = row[pos]

        # Determine range for new value
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < num_elements - 1 else _scope(current) + 5

        if low >= high:
            T *= cooling
            continue

        # Try a random new value in range
        new_val = rng.randint(low, high)
        if new_val == old_val:
            T *= cooling
            continue

        row[pos] = new_val

        # Check if the row is still internally valid (strictly increasing, no dup diffs within row)
        valid_internal = True
        for i in range(len(row) - 1):
            if row[i] >= row[i + 1]:
                valid_internal = False
                break
        if valid_internal:
            diffs = _row_diffs_list(row)
            if len(diffs) != len(set(diffs)):
                valid_internal = False

        if not valid_internal:
            row[pos] = old_val
            T *= cooling
            continue

        new_cost = _sa_total_cost(current, num_elements, counting_bound)
        delta = new_cost - current_cost

        # Accept if better or with probability exp(-delta/T)
        if delta <= 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best = [r[:] for r in current]
        else:
            row[pos] = old_val

        T *= cooling

    return best, best_cost


def _local_search_down(rows, n_rows, num_elements, rng, deadline):
    """
    Deterministic local search: repeatedly try to move elements down.
    """
    start_time = time.time()
    current = [r[:] for r in rows]
    current_scope = _scope(current)
    best = [r[:] for r in current]
    best_scope = current_scope

    row_masks = [_row_mask(r) for r in current]
    global_mask = 0
    for m in row_masks:
        global_mask |= m

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r in range(n_rows):
            for pos in range(num_elements - 1, 0, -1):
                row = current[r]
                old_val = row[pos]
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1 if pos < num_elements - 1 else old_val - 1

                if low >= old_val:
                    continue

                # Try to move down one step at a time
                for new_val in range(old_val - 1, low - 1, -1):
                    row[pos] = new_val

                    # Check internal validity
                    diffs = _row_diffs_list(row)
                    if len(diffs) != len(set(diffs)):
                        continue

                    new_mask = _row_mask(row)
                    other_mask = global_mask & ~row_masks[r]
                    if new_mask & other_mask:
                        continue

                    # Accept
                    row_masks[r] = new_mask
                    global_mask = (global_mask & ~row_masks[r]) | new_mask
                    current_scope = _scope(current)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best = [r[:] for r in current]
                    improved = True
                    break
                else:
                    row[pos] = old_val

    return best, best_scope


def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin

    num_elements = k + 1
    total_diffs_needed = n_rows * (num_elements * (num_elements - 1) // 2)
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))

    best_rows = None
    best_scope = float('inf')

    # Phase 1: Multiple greedy constructions with different seeds
    for seed in range(50):
        if time.time() > DEADLINE - 80:
            break
        local_rng = random.Random(seed * 31 + 7)
        # Try increasing scope hints
        for offset in [0, 3, 6, 10, 15, 25, 40]:
            scope_hint = counting_bound + offset
            rows = _greedy_construct(n_rows, num_elements, local_rng, scope_hint)
            if rows is not None:
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                if s <= counting_bound + 3:
                    break
        if best_scope <= counting_bound + 3:
            break

    # Fallback: simple sequential construction
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)

    # Phase 2: Simulated annealing
    remaining = DEADLINE - time.time()
    if remaining > 60:
        sa_deadline = DEADLINE - 30
        rng = random.Random(12345)
        sa_rows, sa_cost = _simulated_annealing(
            best_rows, n_rows, num_elements, counting_bound, rng, sa_deadline
        )
        if sa_cost == 0:  # Valid solution
            sa_scope = _scope(sa_rows)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows

    # Phase 3: Local search to push scope down
    remaining = DEADLINE - time.time()
    if remaining > 20:
        ls_deadline = DEADLINE - 5
        rng = random.Random(67890)
        ls_rows, ls_scope = _local_search_down(
            best_rows, n_rows, num_elements, rng, ls_deadline
        )
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows

    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
