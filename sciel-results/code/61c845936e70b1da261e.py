import time
import sys
import random

# EVOLVE-BLOCK-START

def _greedy_row(used, k):
    """Build one row greedily: smallest x whose differences to current row are all unused."""
    row = [0]
    x = 0
    row_diffs = set()
    while len(row) < k + 1:
        x += 1
        ok = True
        for a in row:
            d = x - a
            if d in used or d in row_diffs:
                ok = False
                break
        if ok:
            row.append(x)
            row_diffs.update(x - a for a in row[:-1])
            used |= row_diffs
    return row, used


def _backtrack_row(used, k, M, deadline, rng):
    """
    Find a row of k+1 increasing ints starting at 0, max <= M,
    such that all intra-row differences are not in `used`.
    Returns (row, new_used) or (None, used) if not found / deadline hit.
    """
    # We build the row element by element.
    # State: current row list, current max allowed M.
    row = [0]
    row_diffs = set()

    def dfs(depth, last):
        nonlocal row, row_diffs
        if time.time() > deadline:
            return False
        if depth == k + 1:
            return True
        # Try candidate next value
        # Lower bound: last+1
        # Upper bound: M
        start = last + 1
        # Heuristic: try values that create fewer "future conflicts"
        # For simplicity, try in increasing order but with a small randomization
        candidates = list(range(start, M + 1))
        if rng.random() < 0.3:
            # Occasionally shuffle the upper half to diversify
            mid = len(candidates) // 2
            if mid < len(candidates):
                rng.shuffle(candidates[mid:])

        for cand in candidates:
            if time.time() > deadline:
                return False
            # Check all differences cand - a for a in row
            new_diffs = []
            ok = True
            for a in row:
                d = cand - a
                if d in used or d in row_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            if not ok:
                continue
            # Also ensure new_diffs don't conflict among themselves (they can't since row is increasing)
            # Place
            row.append(cand)
            row_diffs.update(new_diffs)
            if dfs(depth + 1, cand):
                return True
            # Undo
            row.pop()
            row_diffs.difference_update(new_diffs)
        return False

    if dfs(1, 0):
        return row, used | row_diffs
    return None, used


def _construct_greedy(n_rows, k):
    """Fast greedy construction, returns (rows, max_val)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, used = _greedy_row(used, k)
        rows.append(row)
    max_val = max(r[-1] for r in rows)
    return rows, max_val


def _construct_search(n_rows, k, deadline, rng):
    """
    Iterative deepening on M (max value). For each M, try to construct all rows
    with max <= M using backtracking. Rows are ordered by 'hardest first' heuristic.
    Returns best valid (rows, max_val) found before deadline.
    """
    # Start with greedy solution
    rows_greedy, max_greedy = _construct_greedy(n_rows, k)
    best_rows = [r[:] for r in rows_greedy]
    best_max = max_greedy

    # Try to improve: search for solutions with smaller max
    # Lower bound: counting bound ~ n_rows * k * (k+1) / 2 differences must fit in [1, M]
    # So M >= n_rows * k * (k+1) / 2
    lb = (n_rows * k * (k + 1)) // 2
    if lb > max_greedy:
        lb = max_greedy  # greedy already better than lb

    # Search from lb up to max_greedy - 1
    # For each M, try to construct all rows with max <= M
    # Use a time budget per M
    total_deadline = deadline
    time_per_M = max(0.5, (total_deadline - time.time()) / max(1, (max_greedy - lb)))

    for M in range(lb, best_max):
        if time.time() > total_deadline:
            break
        M_deadline = min(total_deadline, time.time() + time_per_M)
        
        # Try to construct n_rows rows each with max <= M
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            if time.time() > M_deadline:
                success = False
                break
            row, used = _backtrack_row(used, k, M, M_deadline, rng)
            if row is None:
                success = False
                break
            rows.append(row)
        
        if success and len(rows) == n_rows:
            # Valid solution with max <= M
            actual_max = max(r[-1] for r in rows)
            if actual_max < best_max:
                best_rows = [r[:] for r in rows]
                best_max = actual_max
            # If we found a solution at M, we can stop (M is the max we searched)
            # But actual_max might be < M, so try smaller M
            # Actually, we're iterating M downward in effect? No, upward from lb.
            # If we find a solution at M, try M-1, M-2, ...
            # But our loop goes upward. Let's reverse: try from best_max-1 downward.
            break

    # If the upward loop didn't help, try downward from best_max - 1
    # Actually, let's do a proper downward search
    # Reset: try M from best_max - 1 down to lb
    for M in range(best_max - 1, lb - 1, -1):
        if time.time() > total_deadline:
            break
        M_deadline = min(total_deadline, time.time() + 1.0)  # 1s per M
        
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            if time.time() > M_deadline:
                success = False
                break
            row, used = _backtrack_row(used, k, M, M_deadline, rng)
            if row is None:
                success = False
                break
            rows.append(row)
        
        if success and len(rows) == n_rows:
            actual_max = max(r[-1] for r in rows)
            if actual_max < best_max:
                best_rows = [r[:] for r in rows]
                best_max = actual_max
            # Found a solution at M, try M-1
            # Continue loop to try smaller M
        else:
            # Failed at M, try smaller M
            continue

    return best_rows, best_max


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 150s limit, stop at 145s
    
    # Phase 1: Fast greedy
    rows, max_val = _construct_greedy(n_rows, k)
    
    if time.time() > deadline - 5.0:
        return rows
    
    # Phase 2: Search for better solution
    rng = random.Random(42)
    try:
        best_rows, best_max = _construct_search(n_rows, k, deadline, rng)
        if best_max < max_val:
            rows = best_rows
            max_val = best_max
    except Exception:
        pass  # Keep greedy solution
    
    # Final check: ensure we return before deadline
    if time.time() > deadline:
        # Return whatever we have (should be valid)
        pass
    
    return rows

# EVOLVE-BLOCK-END
