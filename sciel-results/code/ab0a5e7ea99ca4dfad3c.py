"""CSP + randomized local search for difference triangle set (n_rows, k)."""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return list of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    for d in _row_diffs(row):
        m |= (1 << d)
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))


def _greedy_row(max_scope, used_mask, num_elements, rng):
    """
    Greedily build one row of num_elements increasing ints starting at 0,
    with max element <= max_scope, avoiding differences in used_mask.
    Returns row or None.
    """
    for _ in range(300):
        if num_elements - 1 > max_scope:
            return None
        vals = sorted(rng.sample(range(1, max_scope + 1), num_elements - 1))
        row = [0] + vals
        if row[-1] > max_scope:
            continue
        diffs = _row_diffs(row)
        if len(diffs) != len(set(diffs)):
            continue
        m = _row_mask(row)
        if m & used_mask:
            continue
        return row
    return None


def _greedy_construct(n_rows, num_elements, max_scope, rng, max_restarts=50):
    """
    Greedy: build rows one by one, each avoiding used differences.
    """
    for restart in range(max_restarts):
        used_mask = 0
        rows = []
        success = True
        for _ in range(n_rows):
            row = _greedy_row(max_scope, used_mask, num_elements, rng)
            if row is None:
                success = False
                break
            rows.append(row)
            used_mask |= _row_mask(row)
        if success:
            return rows
    return None


def _local_search(rows, num_elements, max_scope, rng, time_limit):
    """
    Local search: try to reduce scope by replacing individual elements.
    Uses a full global validity check after each accepted move to ensure correctness.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope

    n_rows = len(current_rows)

    def compute_global_mask(rows_list):
        gm = 0
        for r in rows_list:
            gm |= _row_mask(r)
        return gm

    def try_move_element(row_idx, pos, new_val):
        """
        Try setting row_idx[pos] = new_val.
        Returns True if valid and applied, False otherwise.
        """
        row = current_rows[row_idx][:]
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False
        if new_val <= 0:
            return False
        row[pos] = new_val
        # Check internal diffs distinct
        diffs = _row_diffs(row)
        if len(diffs) != len(set(diffs)):
            return False
        # Check global validity: all diffs in all rows must be distinct
        # Build a set of all diffs from other rows
        other_diffs = set()
        for i, r in enumerate(current_rows):
            if i != row_idx:
                other_diffs.update(_row_diffs(r))
        # Check new row diffs against other_diffs
        for d in diffs:
            if d in other_diffs:
                return False
        # Apply
        current_rows[row_idx] = row
        return True

    max_iter = 30000
    for it in range(max_iter):
        if time.time() > time_limit:
            break

        # Pick a random row and position
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, len(current_rows[row_idx]) - 1)
        old_val = current_rows[row_idx][pos]

        low = current_rows[row_idx][pos - 1] + 1
        high = current_rows[row_idx][pos + 1] - 1 if pos < len(current_rows[row_idx]) - 1 else max_scope

        if low >= high:
            continue

        # Bias toward smaller values
        if rng.random() < 0.7:
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            new_val = rng.randint(low, high)

        if new_val == old_val:
            continue

        # Try the move
        if try_move_element(row_idx, pos, new_val):
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                max_scope = best_scope
        # If move failed, state is unchanged (try_move_element only modifies on success)

    return best_rows, best_scope


def _local_search_swap(rows, num_elements, max_scope, rng, time_limit):
    """
    Local search variant: swap elements between rows or within a row.
    Uses full global validity check.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    n_rows = len(current_rows)

    def try_swap(row_idx1, pos1, row_idx2, pos2):
        """Swap elements at (row_idx1, pos1) and (row_idx2, pos2)."""
        # Both positions must be >= 1 (index 0 is always 0)
        if pos1 == 0 or pos2 == 0:
            return False
        v1 = current_rows[row_idx1][pos1]
        v2 = current_rows[row_idx2][pos2]
        if v1 == v2:
            return False
        # Try swap
        current_rows[row_idx1][pos1] = v2
        current_rows[row_idx2][pos2] = v1
        # Check both rows are increasing
        r1 = current_rows[row_idx1]
        r2 = current_rows[row_idx2]
        if any(r1[i] >= r1[i+1] for i in range(len(r1)-1)):
            current_rows[row_idx1][pos1] = v1
            current_rows[row_idx2][pos2] = v2
            return False
        if any(r2[i] >= r2[i+1] for i in range(len(r2)-1)):
            current_rows[row_idx1][pos1] = v1
            current_rows[row_idx2][pos2] = v2
            return False
        # Check internal diffs distinct for both rows
        d1 = _row_diffs(r1)
        d2 = _row_diffs(r2)
        if len(d1) != len(set(d1)) or len(d2) != len(set(d2)):
            current_rows[row_idx1][pos1] = v1
            current_rows[row_idx2][pos2] = v2
            return False
        # Check global validity
        all_diffs = []
        for i, r in enumerate(current_rows):
            if i != row_idx1 and i != row_idx2:
                all_diffs.extend(_row_diffs(r))
        all_diffs.extend(d1)
        all_diffs.extend(d2)
        if len(all_diffs) != len(set(all_diffs)):
            current_rows[row_idx1][pos1] = v1
            current_rows[row_idx2][pos2] = v2
            return False
        return True

    max_iter = 20000
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        # Pick two distinct rows
        row_idx1 = rng.randint(0, n_rows - 1)
        row_idx2 = rng.randint(0, n_rows - 1)
        if row_idx1 == row_idx2:
            continue
        pos1 = rng.randint(1, len(current_rows[row_idx1]) - 1)
        pos2 = rng.randint(1, len(current_rows[row_idx2]) - 1)
        if try_swap(row_idx1, pos1, row_idx2, pos2):
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                max_scope = best_scope

    return best_rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs

    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))

    initial_scope = max(counting_bound, 100)
    max_scope_limit = 200

    rng = random.Random(42)

    best_rows = None
    best_scope = float('inf')

    # Phase 1: Greedy construction
    for scope in range(initial_scope, max_scope_limit + 1):
        if time.time() > DEADLINE - 40:
            break
        rows = _greedy_construct(n_rows, num_elements, scope, rng, max_restarts=40)
        if rows is not None:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
            if s <= scope:
                break
        if scope - initial_scope >= 15:
            break

    if best_rows is None:
        for scope in range(max_scope_limit, 300):
            rows = _greedy_construct(n_rows, num_elements, scope, rng, max_restarts=100)
            if rows is not None:
                best_rows = rows
                best_scope = _scope(rows)
                break
        if best_rows is None:
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            best_rows = rows
            best_scope = _scope(rows)

    # Phase 2: Alternating local search (element moves and swaps)
    while time.time() < DEADLINE - 15:
        # Element move local search
        t1 = time.time()
        improved_rows, improved_scope = _local_search(
            best_rows, num_elements, best_scope, rng,
            time_limit=min(DEADLINE - 20, t1 + 20)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
        if best_scope <= counting_bound:
            break
        if time.time() > DEADLINE - 20:
            break
        # Swap local search
        t2 = time.time()
        improved_rows, improved_scope = _local_search_swap(
            best_rows, num_elements, best_scope, rng,
            time_limit=min(DEADLINE - 20, t2 + 15)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
        if best_scope <= counting_bound:
            break

    # Phase 3: Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
