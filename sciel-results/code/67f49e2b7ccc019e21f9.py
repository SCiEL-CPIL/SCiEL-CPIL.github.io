import time
import random
import sys

sys.setrecursionlimit(100000)

DEADLINE = 0
TIME_LIMIT = 115


def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT

    # Always-available valid baseline
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return []

    baseline_scope = max(max(row) for row in baseline_rows)

    # Search for a better (smaller max entry) solution
    generic_lb = n_rows * (k * (k + 1) // 2)
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = generic_lb

    start_m = max(lower_bound, generic_lb)
    best_rows = baseline_rows
    best_scope = baseline_scope

    for M in range(start_m, best_scope + 1):
        if time.time() > DEADLINE:
            break
        sol = solve_with_precompute(n_rows, k, M)
        if sol is not None:
            return sol

    return best_rows


def greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def row_to_mask(row, M):
    """Convert a row to a bitmask of its differences."""
    mask = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d <= M:
                mask |= (1 << d)
    return mask


def precompute_valid_rows(n_rows, k, M, deadline):
    """Precompute all valid rows (starting at 0, k+1 elements, max <= M) 
    and their difference masks."""
    valid_rows = []
    
    def build_row(current_row, current_mask):
        if time.time() > deadline:
            return
        if len(current_row) == k + 1:
            valid_rows.append((list(current_row), current_mask))
            return
        last_val = current_row[-1]
        for x in range(last_val + 1, M + 1):
            new_mask = current_mask
            valid = True
            for a in current_row:
                d = x - a
                if (1 << d) & current_mask:
                    valid = False
                    break
                new_mask |= (1 << d)
            if valid:
                current_row.append(x)
                build_row(current_row, new_mask)
                current_row.pop()
    
    build_row([0], 0)
    return valid_rows


def solve_with_precompute(n_rows, k, M):
    """Solve using precomputed rows and local search for optimal permutation."""
    deadline = DEADLINE - 5  # safety margin
    
    # Precompute all valid rows
    valid_rows = precompute_valid_rows(n_rows, k, M, deadline)
    
    if len(valid_rows) < n_rows:
        return None
    
    if time.time() > deadline:
        return None
    
    # Try to find n_rows compatible rows
    # Strategy: backtracking with bitmask compatibility
    best_rows = None
    best_max = M + 1
    
    # First try deterministic backtracking
    result = backtrack_select(n_rows, k, M, valid_rows, deadline)
    if result is not None:
        return result
    
    # Try randomized restarts
    for attempt in range(20):
        if time.time() > deadline:
            break
        random.shuffle(valid_rows)
        result = backtrack_select(n_rows, k, M, valid_rows, deadline)
        if result is not None:
            return result
    
    return None


def backtrack_select(n_rows, k, M, valid_rows, deadline):
    """Backtracking to select n_rows compatible rows."""
    used_mask = 0
    selected = []
    
    def dfs(idx, used_mask, selected):
        nonlocal used_mask
        if time.time() > deadline:
            return False
        
        if len(selected) == n_rows:
            return True
        
        for row, mask in valid_rows:
            if used_mask & mask:
                continue
            selected.append(row)
            if dfs(idx + 1, used_mask | mask, selected):
                return True
            selected.pop()
        
        return False
    
    if dfs(0, 0, selected):
        return selected
    
    return None


# Fallback to original solve if precompute approach fails
def solve(n_rows, k, M):
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    bit_of = [1 << d for d in range(M + 1)]

    used_mask = 0
    rows = []

    def build_row(current_row, current_row_mask, rows_remaining, randomize_order):
        nonlocal used_mask
        if time.time() > DEADLINE:
            return False

        if len(current_row) == k + 1:
            rows.append(list(current_row))
            used_mask |= current_row_mask
            return True

        remaining_needed = (rows_remaining + 1) * diffs_per_row - current_row_mask.bit_count()
        available = M - used_mask.bit_count()
        if available < remaining_needed:
            return False

        last_val = current_row[-1]
        candidates = []
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                bit = bit_of[d]
                if (used_mask | current_row_mask) & bit:
                    valid = False
                    break
                new_mask |= bit
            if valid:
                candidates.append((x, new_mask))

        if not candidates:
            return False

        if randomize_order:
            random.shuffle(candidates)
        else:
            candidates.sort(key=lambda t: t[0])

        for x, new_mask in candidates:
            current_row.append(x)
            if build_row(current_row, current_row_mask | new_mask, rows_remaining - 1, randomize_order):
                return True
            current_row.pop()

        return False

    def run_search(randomize):
        nonlocal used_mask, rows
        used_mask = 0
        rows = []
        for i in range(n_rows):
            if time.time() > DEADLINE:
                return False
            if not build_row([0], 0, n_rows - i - 1, randomize):
                return False
        return True

    if run_search(False):
        return rows

    for _ in range(12):
        if time.time() > DEADLINE:
            return None
        if run_search(True):
            return rows

    return None
