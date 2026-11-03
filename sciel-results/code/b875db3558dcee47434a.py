"""Row-level difference-set search for the difference-triangle problem."""
import time
import random
import math
from collections import Counter

# EVOLVE-BLOCK-START


def _scope(rows):
    return max(max(r) for r in rows)


def _row_to_diffs(row):
    """Return sorted list of all positive pairwise differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return sorted(diffs)


def _diffs_to_row(diffs, k):
    """
    Reconstruct a row of k+1 elements starting at 0 from a set of pairwise diffs.
    The row has k+1 elements, so there are C(k+1,2) = k(k+1)/2 diffs.
    We need to find a_0=0 < a_1 < ... < a_k such that all a_j - a_i for i<j are in diffs.
    
    Strategy: a_1 must be the smallest diff. Then work forward.
    Returns the row as a list, or None if inconsistent.
    """
    n = k + 1
    target_count = k * (k + 1) // 2
    if len(diffs) != target_count:
        return None
    
    # a[0] = 0
    row = [0]
    used_diffs = set()
    
    # The smallest diff must be a[1] - a[0] = a[1]
    if not diffs:
        return None
    a1 = diffs[0]
    row.append(a1)
    used_diffs.add(a1)
    
    # Now we need to find a[2], ..., a[k]
    # a[j] - a[i] for all i < j must be in diffs
    # Key insight: a[j] = a[j-1] + d where d is some diff
    # But more directly: for each new element a[j], the diffs a[j]-a[0], a[j]-a[1], ..., a[j]-a[j-1]
    # must all be in the remaining diffs set.
    
    remaining = list(diffs)
    remaining.remove(a1)
    
    for j in range(2, n):
        # a[j] must satisfy: a[j] - a[i] is in remaining for all i in 0..j-1
        # And a[j] - a[j-1] > 0 (strictly increasing)
        # Candidate: a[j] = a[j-1] + d for some d, but more generally
        # a[j] - a[0] = a[j] must be in remaining
        # a[j] - a[1] must be in remaining
        # etc.
        
        # Try: a[j] must be such that all (a[j] - a[i]) for i in range(j) are in remaining
        # The smallest possible a[j] is a[j-1] + 1
        # But we can be smarter: a[j] - a[0] = a[j] must be in remaining
        # So a[j] is a candidate value from remaining (since a[j] - 0 = a[j])
        
        found = False
        # Candidates for a[j]: values v in remaining such that v - a[i] is also in remaining for all i < j
        # And v > a[j-1]
        candidates = []
        rem_set = set(remaining)
        for v in rem_set:
            if v <= row[-1]:
                continue
            # Check all differences
            valid = True
            for i in range(j):
                d = v - row[i]
                if d not in rem_set:
                    valid = False
                    break
            if valid:
                candidates.append(v)
        
        if not candidates:
            return None
        
        # Pick the smallest valid candidate (greedy for compactness)
        # Actually, we should try to be flexible. Let's pick the smallest.
        a_j = min(candidates)
        row.append(a_j)
        
        # Remove the diffs a_j - a[i] for i in range(j) from remaining
        for i in range(j):
            d = a_j - row[i]
            # Remove one instance
            idx = remaining.index(d)
            remaining.pop(idx)
    
    # Verify: row should have n elements and be strictly increasing
    if len(row) != n:
        return None
    for i in range(1, n):
        if row[i] <= row[i - 1]:
            return None
    
    # Final verification: all pairwise diffs match
    actual_diffs = _row_to_diffs(row)
    if sorted(actual_diffs) != sorted(diffs):
        return None
    
    return row


def _fast_greedy(n_rows, k):
    """Greedy construction: pick each row to minimize its max given used diffs."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = [x - a for a in row]
            # Check internal distinctness
            if len(set(new_diffs)) != len(new_diffs):
                continue
            # Check against used
            if any(d in used for d in new_diffs):
                continue
            row.append(x)
            used.update(new_diffs)
        rows.append(row)
    return rows


def _build_global_diff_set(rows):
    """Build set of all diffs across all rows."""
    used = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                used.add(r[j] - r[i])
    return used


def _row_diffs(row):
    """Get set of diffs for a single row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _sa_row_level(rows, n_rows, k, time_limit, rng):
    """
    Simulated annealing operating at the row-difference-set level.
    Each move: pick a row, pick a new Sidon set of diffs (same size), 
    reconstruct the row, and check global validity.
    """
    row_len = k + 1
    n_diffs_per_row = k * (k + 1) // 2
    
    rows = [list(r) for r in rows]
    global_used = _build_global_diff_set(rows)
    row_diff_sets = [_row_diffs(r) for r in rows]
    
    current_max = max(r[-1] for r in rows)
    current_collisions = 0  # should be 0 for valid input
    current_cost = current_max
    
    best_cost = current_cost
    best_rows = [list(r) for r in rows]
    best_max = current_max
    
    T = 30.0
    cooling = 0.9997
    start_time = time.time()
    
    def _reconstruct_from_diffs(diffs_list, k):
        """Reconstruct row from sorted diff list. Returns row or None."""
        n = k + 1
        if len(diffs_list) != k * (k + 1) // 2:
            return None
        if len(set(diffs_list)) != len(diffs_list):
            return None  # duplicate diffs within row
        
        row = [0]
        remaining = list(diffs_list)
        
        # a[1] = smallest diff
        a1 = remaining[0]
        row.append(a1)
        remaining.pop(0)
        
        for j in range(2, n):
            rem_set = set(remaining)
            best_candidate = None
            for v in rem_set:
                if v <= row[-1]:
                    continue
                valid = True
                for i in range(j):
                    if (v - row[i]) not in rem_set:
                        valid = False
                        break
                if valid:
                    if best_candidate is None or v < best_candidate:
                        best_candidate = v
                    # We want the smallest valid one for compactness
                    # But for SA diversity, maybe random? Let's use smallest for now.
                    break
            
            if best_candidate is None:
                return None
            
            row.append(best_candidate)
            for i in range(j):
                d = best_candidate - row[i]
                idx = remaining.index(d)
                remaining.pop(idx)
        
        # Verify
        if len(row) != n:
            return None
        for i in range(1, n):
            if row[i] <= row[i - 1]:
                return None
        
        # Full verification
        actual = _row_to_diffs(row)
        if sorted(actual) != sorted(diffs_list):
            return None
        
        return row
    
    def _generate_neighbor_diffs(current_diffs, scope_limit):
        """
        Generate a neighboring Sidon set of diffs by perturbing the current set.
        Returns a new sorted list of diffs or None.
        """
        n_diffs = len(current_diffs)
        new_diffs = list(current_diffs)
        
        # Pick 1-3 elements to modify
        n_mods = rng.randint(1, 3)
        for _ in range(n_mods):
            idx = rng.randint(0, n_diffs - 1)
            old_val = new_diffs[idx]
            # Try a new value
            # Bias toward smaller values to reduce scope
            if rng.random() < 0.6:
                # Try smaller
                new_val = max(1, old_val - rng.randint(1, 15))
            else:
                # Try larger (within scope)
                new_val = old_val + rng.randint(1, 10)
            
            if new_val == old_val:
                continue
            if new_val < 1:
                continue
            if new_val > scope_limit:
                continue
            # Check no duplicates
            if new_val in new_diffs:
                continue
            new_diffs[idx] = new_val
        
        new_diffs.sort()
        
        # Check it's a valid Sidon set (all distinct)
        if len(set(new_diffs)) != len(new_diffs):
            return None
        
        return new_diffs
    
    while time.time() - start_time < time_limit:
        r_idx = rng.randint(0, n_rows - 1)
        
        # Get current diffs for this row
        cur_diffs = sorted(row_diff_sets[r_idx])
        
        # Determine scope limit: current global max + some margin
        scope_limit = best_max + 20
        
        # Generate neighbor diff set
        new_diffs = _generate_neighbor_diffs(cur_diffs, scope_limit)
        if new_diffs is None:
            continue
        
        # Reconstruct row from new diffs
        new_row = _reconstruct_from_diffs(new_diffs, k)
        if new_row is None:
            continue
        
        # Check global validity: new row's diffs must not conflict with other rows' diffs
        new_row_diffs = _row_diffs(new_row)
        other_diffs = set()
        for i in range(n_rows):
            if i != r_idx:
                other_diffs |= row_diff_sets[i]
        
        if new_row_diffs & other_diffs:
            # Conflict - reject
            continue
        
        # Compute new cost
        new_max = max(max(r) for r in rows if r is not new_row) if False else 0
        # Actually compute properly
        new_max = 0
        for i in range(n_rows):
            if i == r_idx:
                new_max = max(new_max, new_row[-1])
            else:
                new_max = max(new_max, rows[i][-1])
        
        new_cost = new_max
        
        # Accept/reject
        if new_cost < current_cost:
            accept = True
        else:
            delta = new_cost - current_cost
            if T > 0.1:
                accept = rng.random() < math.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            # Update state
            old_diffs = row_diff_sets[r_idx]
            rows[r_idx] = new_row
            row_diff_sets[r_idx] = new_row_diffs
            current_cost = new_cost
            current_max = new_max
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [list(r) for r in rows]
                best_max = current_max
        else:
            # No state change needed since we didn't modify in-place
            pass

        T *= cooling
    
    return best_rows, best_max


def _local_compression(rows, k, deadline, rng):
    """
    Local search: for each row, try to compress by finding a smaller valid 
    diff set that doesn't conflict with other rows.
    """
    best_rows = [list(r) for r in rows]
    best_sc = _scope(best_rows)
    n_rows = len(best_rows)
    
    # Precompute other rows' diffs
    def _get_other_diffs(ri):
        s = set()
        for i in range(n_rows):
            if i != ri:
                s |= _row_diffs(best_rows[i])
        return s
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(n_rows):
            row = best_rows[ri]
            other_diffs = _get_other_diffs(ri)
            
            # Try to compress: for each position from last to first, try smaller values
            for pos in range(len(row) - 1, 0, -1):
                lower = row[pos - 1] + 1
                upper = row[pos] - 1
                if upper < lower:
                    continue
                
                # Try candidates from lower upward
                for cand in range(lower, upper + 1):
                    if cand == row[pos]:
                        break
                    
                    old = row[pos]
                    row[pos] = cand
                    
                    # Check validity
                    new_row_diffs = _row_diffs(row)
                    
                    # Check internal distinctness
                    if len(new_row_diffs) != len(row) * (len(row) - 1) // 2:
                        row[pos] = old
                        break
                    
                    # Check against other rows
                    if new_row_diffs & other_diffs:
                        row[pos] = old
                        break
                    
                    # Valid! Update
                    new_sc = _scope(best_rows)
                    if new_sc < best_sc:
                        best_sc = new_sc
                        improved = True
                    else:
                        row[pos] = old
                        break
                else:
                    # No improvement found at this position
                    pass
        
        # Also try swapping rows or reordering
        if time.time() < deadline:
            # Try: for each row, try to find an entirely new compressed version
            for ri in range(n_rows):
                if time.time() >= deadline:
                    break
                row = best_rows[ri]
                other_diffs = _get_other_diffs(ri)
                
                # Try greedy rebuild with tighter bound
                current_max = row[-1]
                # Try to build a new row with max < current_max
                for target_max in range(current_max - 1, 0, -1):
                    if target_max < len(row) - 1:
                        break
                    # Greedy build
                    new_row = [0]
                    x = 0
                    success = True
                    while len(new_row) < len(row):
                        x += 1
                        if x > target_max:
                            success = False
                            break
                        new_diffs = [x - a for a in new_row]
                        if len(set(new_diffs)) != len(new_diffs):
                            continue
                        if any(d in other_diffs for d in new_diffs):
                            continue
                        # Also check against already-placed new_row diffs
                        existing = set()
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                existing.add(new_row[j] - new_row[i])
                        if any(d in existing for d in new_diffs):
                            continue
                        new_row.append(x)
                    
                    if success and len(new_row) == len(row):
                        new_row_diffs = _row_diffs(new_row)
                        if not (new_row_diffs & other_diffs):
                            # Check internal
                            if len(new_row_diffs) == len(row) * (len(row) - 1) // 2:
                                best_rows[ri] = new_row
                                new_sc = _scope(best_rows)
                                if new_sc < best_sc:
                                    best_sc = new_sc
                                    improved = True
                                break
    
    return best_rows, best_sc


def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = random.Random(12345)
    
    # Phase 1: Fast greedy baseline
    best_rows = _fast_greedy(n_rows, k)
    best_sc = _scope(best_rows)
    
    # Phase 2: Row-level simulated annealing
    sa_time = min(90.0, deadline - 30.0)
    if time.time() < deadline - 30.0:
        rows, sc = _sa_row_level(best_rows, n_rows, k, sa_time, rng)
        if sc < best_sc:
            best_sc = sc
            best_rows = rows
    
    # Phase 3: Local compression to polish
    if time.time() < deadline - 5.0:
        rows, sc = _local_compression(best_rows, k, deadline - 5.0, rng)
        if sc < best_sc:
            best_sc = sc
            best_rows = rows
    
    # Final validation
    all_diffs = []
    for r in best_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                all_diffs.append(r[j] - r[i])
    
    if len(all_diffs) != len(set(all_diffs)):
        # Invalid - fall back to greedy
        best_rows = _fast_greedy(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
