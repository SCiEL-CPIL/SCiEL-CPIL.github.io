"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, seed=42):
    """
    Build initial solution greedily.
    """
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

def _validate(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        # Check Sidon
        diff_list = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                if d in diff_list:
                    return False
                diff_list.append(d)
                used.add(d)
    return True

def _compute_all_used(rows):
    """Compute the union of all difference sets across all rows."""
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(r[j] - r[i])
    return used

def _global_max(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _max_except(rows, exclude_idx):
    """Return the maximum entry across all rows except the excluded one."""
    m = 0
    for i, r in enumerate(rows):
        if i != exclude_idx:
            if r[-1] > m:
                m = r[-1]
    return m

def _row_diffs_set(row):
    """Compute the set of all pairwise differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _find_random_valid_row(forbidden, max_limit, k, rng, max_candidates=10):
    """
    Construct a random valid row of length k+1 starting at 0, strictly increasing,
    with all internal differences distinct and disjoint from forbidden,
    with max entry <= max_limit.
    """
    row = [0]
    used_in_row = set()
    
    for _ in range(k):
        candidates = []
        start = row[-1] + 1
        end = min(max_limit, start + 400) if max_limit is not None else start + 400
        
        if end < start:
            return None
        
        # Scan for valid candidates
        for x in range(start, end):
            new_diffs = {x - a for a in row}
            if new_diffs & forbidden:
                continue
            if new_diffs & used_in_row:
                continue
            candidates.append(x)
            if len(candidates) >= max_candidates:
                break
        
        if not candidates:
            return None
        
        # Pick a candidate with bias towards smaller values
        # Use a power-law distribution to favor smaller values
        if len(candidates) == 1:
            chosen_x = candidates[0]
        else:
            # Weight: w_i = 1 / (i + 1)^0.5 for i-th candidate (0-indexed)
            weights = [1.0 / math.sqrt(i + 1) for i in range(len(candidates))]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumsum = 0
            chosen_x = candidates[-1]
            for i, w in enumerate(weights):
                cumsum += w
                if r <= cumsum:
                    chosen_x = candidates[i]
                    break
        
        row.append(chosen_x)
        for a in row[:-1]:
            used_in_row.add(chosen_x - a)
        
    return row

def _find_random_valid_row_bounded(forbidden, min_limit, max_limit, k, rng, max_candidates=10):
    """
    Construct a random valid row with max entry in [min_limit, max_limit].
    """
    if min_limit > max_limit:
        return None
    # First try to find a row with max <= max_limit
    row = _find_random_valid_row(forbidden, max_limit, k, rng, max_candidates)
    if row is None:
        return None
    # Check if max is at least min_limit (it should be since it's increasing from 0)
    if row[-1] < min_limit:
        return None
    return row

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Initialize with greedy
    rows = _greedy_init(n_rows, k, seed=42)
    best = [r[:] for r in rows]
    best_val = _global_max(best)
    best_used = _compute_all_used(best)
    
    # Current state
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_row_diffs = [_row_diffs_set(r) for r in cur]
    cur_used = set(best_used)

    # Simulated Annealing parameters
    T = 8.0
    T_min = 0.05
    cool = 0.997
    
    no_improve_steps = 0
    max_no_improve = 80000
    rng = random.Random(12345)

    iteration = 0
    while time.time() < deadline and no_improve_steps < max_no_improve:
        iteration += 1
        # Pick a random row to replace
        ri = rng.randrange(n_rows)
        
        # Calculate forbidden diffs for this row
        forbidden = cur_used - cur_row_diffs[ri]
        
        # Try to find a new row with max <= cur_val (improvement or neutral)
        new_row = _find_random_valid_row(forbidden, cur_val, k, rng, max_candidates=12)
        
        if new_row is None:
            # Try with a slightly larger limit to allow moves that don't immediately improve
            new_row = _find_random_valid_row(forbidden, cur_val + 8, k, rng, max_candidates=12)
        
        if new_row is None:
            no_improve_steps += 1
            T = max(T_min, T * cool)
            continue
            
        # Efficient max calculation
        other_max = _max_except(cur, ri)
        new_max = max(other_max, new_row[-1])
        delta_e = new_max - cur_val

        accept = False
        if delta_e < 0:
            accept = True
        elif delta_e == 0:
            if rng.random() < 0.4:
                accept = True
        else:
            if T > T_min:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            # Remove old diffs BEFORE updating
            old_diffs = cur_row_diffs[ri]
            cur_used -= old_diffs
            new_diffs = _row_diffs_set(new_row)
            cur_used |= new_diffs
            cur_row_diffs[ri] = new_diffs
            
            cur[ri] = new_row
            cur_val = new_max
            
            if cur_val < best_val:
                best_val = cur_val
                best = [r[:] for r in cur]
                best_used = set(cur_used)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1

        T = max(T_min, T * cool)

        # Periodic restart from best
        if no_improve_steps > 20000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set(best_used)
            cur_row_diffs = [_row_diffs_set(r) for r in cur]
            T = 8.0
            no_improve_steps = 0

    # Final polish: try to rebuild each row to minimize its max
    for polish_iter in range(10):
        if time.time() > deadline - 5:
            break
        for ri in range(n_rows):
            if time.time() > deadline - 2:
                break
            forbidden = best_used - _row_diffs_set(best[ri])
            # Try to find a row with max < best_val
            new_row = _find_random_valid_row(forbidden, best_val - 1, k, rng, max_candidates=20)
            if new_row is not None:
                other_max = _max_except(best, ri)
                new_max = max(other_max, new_row[-1])
                if new_max < best_val:
                    best[ri] = new_row
                    best_val = new_max
                    best_used = _compute_all_used(best)

    # Validate best
    if not _validate(best, n_rows, k):
        best = _greedy_init(n_rows, k, seed=42)

    return best

# EVOLVE-BLOCK-END
