"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    out = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def _check_row_valid(row, k):
    """Check if a single row is strictly increasing and starts at 0."""
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    return True

def _try_move(cur, ri, pi, new_val, cur_used, cur_row_diffs, k, n_rows):
    """
    Try to move position pi in row ri to new_val.
    Returns (cand_row, new_diffs, new_max) if valid, else None.
    """
    # Check strict increase within row
    if new_val <= cur[ri][pi - 1]:
        return None
    if pi < k and new_val >= cur[ri][pi + 1]:
        return None
    
    # Build candidate row
    cand_row = cur[ri][:]
    cand_row[pi] = new_val
    
    # Check row validity
    if not _check_row_valid(cand_row, k):
        return None
    
    # Compute new diffs for this row
    new_diffs = _row_diffs(cand_row)
    
    # Check for conflicts with other rows' diffs
    other_diffs = cur_used - cur_row_diffs[ri]
    if new_diffs & other_diffs:
        return None
    
    # Check internal uniqueness in new_diffs (Sidon set property)
    diff_list = _row_diffs_list(cand_row)
    if len(diff_list) != len(set(diff_list)):
        return None
    
    # Compute new max
    cand_max = max(max(r) for r in cur if r is not cand_row) if n_rows > 1 else 0
    cand_max = max(cand_max, max(cand_row))
    
    return (cand_row, new_diffs, cand_max)

def _build_exact_cover(n_rows, k, max_val):
    """
    Build exact cover system for difference triangle set.
    We select n_rows Sidon sets of size k+1 starting at 0,
    such that all differences (across all rows) are distinct and in [1, max_val].
    
    Returns:
        columns: list of column sets (each column is a set of row indices that cover it)
        num_cols: number of columns
    """
    # Each column represents either:
    # 1. A difference value d in [1, max_val] that must be covered exactly once
    # 2. A "row selection" constraint - we need exactly n_rows rows
    
    # Actually, let's model it differently.
    # We have candidate rows (Sidon sets of size k+1 starting at 0 with max <= max_val).
    # Each candidate row covers certain differences.
    # We need to select n_rows candidate rows such that:
    # - All differences covered are distinct (no overlap)
    # - All differences in [1, max_val] are covered? No, we just need them to be distinct.
    # 
    # Wait, the problem says "all differences between two entries of the same row are distinct over all rows".
    # It does NOT require that all differences in [1, max_val] are covered.
    # So this is NOT an exact cover problem in the traditional sense.
    # 
    # Let me reconsider. We want to minimize the maximum entry.
    # The differences must be all distinct across all rows.
    # 
    # For a given max_val M, we need to find n_rows Sidon sets of size k+1 starting at 0
    # with all elements <= M, such that all pairwise differences are distinct.
    # 
    # This is a constraint satisfaction problem, not exact cover.
    # Let me use a different approach: backtracking with constraint propagation.
    
    pass  # We'll use a different method below

def _backtrack_construct(n_rows, k, max_val, time_limit):
    """
    Backtracking search to find n_rows Sidon sets of size k+1 starting at 0
    with all elements <= max_val and all differences distinct.
    """
    start_time = time.time()
    
    # Generate all valid Sidon rows of size k+1 starting at 0 with max <= max_val
    # A Sidon set has all pairwise differences distinct.
    
    def is_sidon(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def gen_rows(limit, current, start_idx):
        """Generate all Sidon rows of size k+1 starting at 0 with max <= limit."""
        if len(current) == k + 1:
            if current[-1] <= limit:
                yield current[:]
            return
        if len(current) == 0:
            current.append(0)
        
        next_start = current[-1] + 1
        for v in range(next_start, limit + 1):
            # Pruning: if we add v, can we still complete to k+1 elements <= limit?
            # We need (k+1 - len(current)) more elements, each at least 1 apart from previous
            # So we need v + (k+1 - len(current) - 1) <= limit
            if v + (k + 1 - len(current) - 1) > limit:
                break
            current.append(v)
            # Check Sidon property incrementally
            ok = True
            new_diffs = set()
            for a in current[:-1]:
                d = v - a
                if d in new_diffs or d in _all_diffs_so_far:
                    ok = False
                    break
                new_diffs.add(d)
            if ok:
                yield from gen_rows(limit, current, v)
            current.pop()
    
    # This approach is too slow. Let's use a different strategy.
    # We'll use randomized local search with a better initialization.
    return None

def _simulated_annealing_improved(n_rows, k, time_limit):
    """
    Improved simulated annealing with:
    1. Better initialization using greedy + random restarts
    2. Multiple restarts with different random seeds
    3. Adaptive cooling
    4. Focus on reducing the maximum entry
    """
    start_time = time.time()
    deadline = start_time + time_limit - 5  # safety margin
    
    best_val = float('inf')
    best_rows = None
    best_used = None
    
    # Try multiple random initializations
    num_initializations = 0
    max_initializations = 20
    
    while time.time() < deadline and num_initializations < max_initializations:
        num_initializations += 1
        random.seed(42 + num_initializations * 137)
        
        # Generate a random initial solution
        rows = _build_greedy(n_rows, k)
        
        # Apply some random perturbations to diversify
        for _ in range(50):
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = rows[ri][pi]
            # Try to move to a random valid position
            for _ in range(10):
                new_val = random.randint(rows[ri][pi-1] + 1, old_val + 20 if pi < k else old_val + 50)
                if pi < k and new_val >= rows[ri][pi + 1]:
                    continue
                cand_row = rows[ri][:]
                cand_row[pi] = new_val
                if not _check_row_valid(cand_row, k):
                    continue
                new_diffs = _row_diffs(cand_row)
                # Check for conflicts
                other_diffs = set()
                for j, r in enumerate(rows):
                    if j != ri:
                        other_diffs |= _row_diffs(r)
                if new_diffs & other_diffs:
                    continue
                # Check Sidon
                diff_list = _row_diffs_list(cand_row)
                if len(diff_list) != len(set(diff_list)):
                    continue
                rows[ri] = cand_row
                break
        
        if not _valid(rows, n_rows, k):
            continue
        
        cur_val = _max_entry(rows)
        if cur_val < best_val:
            best_val = cur_val
            best_rows = [r[:] for r in rows]
            best_used = set()
            for r in best_rows:
                best_used |= _row_diffs(r)
        
        # Run simulated annealing from this starting point
        cur = [r[:] for r in rows]
        cur_val = _max_entry(cur)
        cur_used = set()
        for r in cur:
            cur_used |= _row_diffs(r)
        cur_row_diffs = [_row_diffs(r) for r in cur]
        
        T = 20.0
        T_min = 0.05
        cool = 0.9999
        
        no_improve = 0
        max_no_improve = 10000
        
        while time.time() < deadline and no_improve < max_no_improve:
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = cur[ri][pi]
            
            # Generate candidates
            candidates = []
            for d in range(1, 20):
                nv = old_val - d
                if nv <= cur[ri][pi - 1]:
                    break
                candidates.append(nv)
            for d in range(1, 10):
                nv = old_val + d
                if pi < k and nv >= cur[ri][pi + 1]:
                    break
                candidates.append(nv)
            
            if random.random() < 0.2:
                for d in range(20, 40):
                    nv = old_val - d
                    if nv > cur[ri][pi - 1]:
                        candidates.append(nv)
            
            random.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                result = _try_move(cur, ri, pi, new_val, cur_used, cur_row_diffs, k, n_rows)
                if result is None:
                    continue
                
                cand_row, new_diffs, cand_max = result
                delta_e = cand_max - cur_val
                
                accept = False
                if delta_e < 0:
                    accept = True
                elif delta_e == 0:
                    if random.random() < 0.3:
                        accept = True
                else:
                    if T > T_min:
                        prob = 1.0 / (1.0 + delta_e / max(T, 0.01))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    cur_used -= cur_row_diffs[ri]
                    cur_used |= new_diffs
                    cur[ri] = cand_row
                    cur_row_diffs[ri] = new_diffs
                    cur_val = cand_max
                    
                    if cur_val < best_val:
                        best_val = cur_val
                        best_rows = [r[:] for r in cur]
                        best_used = set(cur_used)
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
            
            T = max(T_min, T * cool)
            
            if no_improve > 5000:
                cur = [r[:] for r in best_rows] if best_rows else _build_greedy(n_rows, k)
                cur_val = _max_entry(cur)
                cur_used = set()
                for r in cur:
                    cur_used |= _row_diffs(r)
                cur_row_diffs = [_row_diffs(r) for r in cur]
                T = 20.0
                no_improve = 0
    
    if best_rows is None:
        best_rows = _build_greedy(n_rows, k)
    
    if not _valid(best_rows, n_rows, k):
        best_rows = _build_greedy(n_rows, k)
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Use improved simulated annealing
    result = _simulated_annealing_improved(n_rows, k, 110)
    
    return result
# EVOLVE-BLOCK-END
