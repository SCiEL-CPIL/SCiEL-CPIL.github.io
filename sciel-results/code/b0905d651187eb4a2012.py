import time
import random
import itertools

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Quick deterministic baseline via smallest-first greedy
    init_rows = _simple_greedy(n_rows, k)
    if not init_rows:
        return []
    current_best_scope = max(max(r) for r in init_rows)
    current_best_rows = init_rows

    # Multi-start randomized construction with interval-set representation
    best_scope = current_best_scope
    best_rows = current_best_rows

    # Budget: use most of the time for multi-start
    rng = random.Random(12345)
    attempt = 0
    while time.time() < deadline - 20.0:
        attempt += 1
        rows = _randomized_interval_construct(n_rows, k, rng)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                # Polish immediately
                if time.time() < deadline - 10.0:
                    polished_rows, polished_scope = _polish(best_rows, k, deadline - 5.0)
                    if polished_scope < best_scope:
                        best_scope = polished_scope
                        best_rows = polished_rows

    # Final polish with remaining time
    if time.time() < deadline - 5.0:
        polished_rows, polished_scope = _polish(best_rows, k, deadline - 2.0)
        if polished_scope < best_scope:
            best_scope = polished_scope
            best_rows = polished_rows

    return best_rows


def _simple_greedy(n_rows, k):
    """Deterministic smallest-first greedy."""
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
                used.update(new_diffs)
        rows.append(row)
    return rows


def _randomized_interval_construct(n_rows, k, rng):
    """
    Construct rows by choosing k distinct positive differences per row that can
    be packed into an increasing sequence starting at 0. The key insight: a row
    with k+1 elements has k 'gap' values (consecutive differences) that sum to
    the row's max, and all pairwise sums of consecutive gaps (i.e., all a_j - a_i)
    must be distinct globally.
    
    We model each row by its k gap values g_1, ..., g_k where element i = sum(g_1..g_i).
    The pairwise differences are all sums of consecutive gaps.
    """
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        row = None
        # Try multiple random attempts for this row
        for trial in range(200):
            # Generate k random positive gaps
            # Bias toward smaller gaps to keep scope tight
            gaps = []
            for gi in range(k):
                # Sample from a distribution that favors small values
                # but ensures we can still reach a reasonable total
                base = rng.randint(1, 3)
                if gi < k // 2:
                    gap = base + rng.randint(0, 2)
                else:
                    gap = base + rng.randint(0, 4)
                gaps.append(gap)
            
            # Build the row from gaps
            row_candidate = [0]
            for g in gaps:
                row_candidate.append(row_candidate[-1] + g)
            
            # Check all pairwise differences
            row_diffs = set()
            valid = True
            for i in range(len(row_candidate)):
                for j in range(i + 1, len(row_candidate)):
                    d = row_candidate[j] - row_candidate[i]
                    if d in row_diffs or d in used_diffs:
                        valid = False
                        break
                    row_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                used_diffs.update(row_diffs)
                rows.append(row_candidate)
                row = row_candidate
                break
        
        if row is None:
            # This row couldn't be constructed; try a different strategy
            # Use a more systematic approach for this row
            row = _greedy_single_row(used_diffs, k, rng)
            if row is None:
                return None
            rows.append(row)
    
    # Verify global validity
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)
    
    return rows


def _greedy_single_row(used_diffs, k, rng):
    """Greedy construction of a single row given already-used differences."""
    row = [0]
    x = 0
    max_trials = 5000
    for _ in range(max_trials):
        x += 1
        if x > 200:  # Give up if we're going too far
            return None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used_diffs):
            row.append(x)
            if len(row) == k + 1:
                used_diffs.update(new_diffs)
                return row
    return None


def _polish(rows, k, deadline):
    """
    Local search: try to reduce the scope by moving elements to smaller values.
    Uses a full global validity check to ensure correctness.
    """
    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)

    def _is_valid(rows):
        all_d = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in all_d:
                        return False
                    all_d.add(d)
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to reduce the scope by moving elements in rows with large values
        # Process rows in order of their max value (largest first)
        row_order = sorted(range(len(best)), key=lambda i: max(best[i]), reverse=True)
        
        for row_idx in row_order:
            if time.time() > deadline:
                break
            row = best[row_idx]
            
            # Try moving each element (except the first, which is 0) to a smaller value
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                
                # Try values from row[pos-1]+1 to old_val-1
                for new_val in range(row[pos - 1] + 1, old_val):
                    row[pos] = new_val
                    
                    # Check validity
                    if _is_valid(best):
                        new_scope = max(max(r) for r in best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            improved = True
                            break
                        else:
                            # Revert
                            row[pos] = old_val
                    else:
                        row[pos] = old_val
                
                if improved:
                    break
            
            if improved:
                # Restart the loop to re-evaluate all rows
                break

    return best, best_scope
