"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Return the set of all pairwise differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    """Return the list of all pairwise differences in a row."""
    n = len(row)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def _is_sidon(row):
    """Check if a row is a Sidon set (all pairwise differences distinct)."""
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return len(diffs) == len(set(diffs))

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        if not _is_sidon(row):
            return False
        for d in _row_diffs_list(row):
            if d in used:
                return False
            used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

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
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _construct_with_max_constraint(n_rows, k, max_val):
    """
    Try to construct n_rows rows of k+1 increasing integers starting at 0
    such that all differences are globally distinct and all entries <= max_val.
    Uses randomized backtracking.
    Returns (rows, success) where rows is the list if successful.
    """
    # Total number of differences needed
    total_diffs = n_rows * (k * (k + 1) // 2)
    # Available differences: 1 to max_val
    if total_diffs > max_val:
        return None, False
    
    random.seed(int(time.time() * 1000) % (2**31))
    
    # We'll build rows one by one
    # For each row, we need to choose k integers from 1..max_val such that
    # the row is a Sidon set and its differences don't conflict with used diffs
    
    used_diffs = set()
    rows = []
    
    def backtrack_row(idx, current_row, used_so_far):
        """Try to complete row idx given current_row (partial)."""
        if len(current_row) == k + 1:
            # Check if this row is Sidon
            diffs = _row_diffs_list(current_row)
            if len(diffs) != len(set(diffs)):
                return False
            # Check no conflict with used_so_far
            if set(diffs) & used_so_far:
                return False
            return True
        
        if idx >= n_rows:
            return False
        
        # Determine the range for the next element
        start = current_row[-1] + 1
        # We need to leave room for remaining elements
        remaining = k + 1 - len(current_row)
        if start + remaining - 1 > max_val:
            return False
        
        # Try values in random order
        candidates = list(range(start, max_val + 1))
        random.shuffle(candidates)
        
        for val in candidates:
            # Quick check: val - current_row[0] must not be in used_so_far
            # and must not conflict with existing diffs in current_row
            new_diffs = {val - a for a in current_row}
            if new_diffs & used_so_far:
                continue
            # Check internal conflicts with current_row's existing diffs
            existing_diffs = set()
            n = len(current_row)
            for i in range(n):
                for j in range(i + 1, n):
                    existing_diffs.add(current_row[j] - current_row[i])
            if new_diffs & existing_diffs:
                continue
            
            # Also check that val - a for different a don't collide with each other
            # (they can't since a are distinct)
            
            current_row.append(val)
            if backtrack_row(idx, current_row, used_so_far):
                return True
            current_row.pop()
        
        return False
    
    # Build rows one by one
    for r in range(n_rows):
        row = [0]
        # Use randomized greedy with some backtracking
        found = False
        for attempt in range(100):
            row = [0]
            x = 0
            success = True
            while len(row) < k + 1:
                x += 1
                if x > max_val:
                    success = False
                    break
                new = {x - a for a in row}
                if not (new & used_diffs):
                    # Check Sidon property for the partial row
                    partial_diffs = []
                    n = len(row)
                    for i in range(n):
                        for j in range(i + 1, n):
                            partial_diffs.append(row[j] - row[i])
                    all_diffs = partial_diffs + list(new)
                    if len(all_diffs) != len(set(all_diffs)):
                        continue
                    row.append(x)
                    used_diffs |= new
                else:
                    # Try a few more values
                    if x > max_val:
                        success = False
                        break
            if success and len(row) == k + 1:
                rows.append(row)
                found = True
                break
        
        if not found:
            return None, False
    
    return rows, True

def _local_search(rows, n_rows, k, deadline):
    """
    Local search to minimize the maximum entry.
    Uses simulated annealing with focused moves.
    """
    start_time = time.time()
    random.seed(42)
    
    best = [r[:] for r in rows]
    best_val = _max_entry(best)
    best_used = set()
    for r in best:
        best_used |= _row_diffs_set(r)
    
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_used = set(best_used)
    cur_row_diffs = [_row_diffs_set(r) for r in cur]
    
    T = 20.0
    T_min = 0.1
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        old_val = cur[ri][pi]
        
        # Generate candidates
        candidates = []
        # Try to reduce values, especially near the max
        for d in range(1, 15):
            nv = old_val - d
            if nv <= cur[ri][pi - 1]:
                break
            candidates.append(nv)
        for d in range(1, 8):
            nv = old_val + d
            if pi < k and nv >= cur[ri][pi + 1]:
                break
            candidates.append(nv)
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= cur[ri][pi - 1]:
                continue
            if pi < k and new_val >= cur[ri][pi + 1]:
                continue
            
            cand_row = cur[ri][:]
            cand_row[pi] = new_val
            
            # Check Sidon property
            if not _is_sidon(cand_row):
                continue
            
            new_diffs = _row_diffs_set(cand_row)
            other_diffs = cur_used - cur_row_diffs[ri]
            if new_diffs & other_diffs:
                continue
            
            cand_max = max(max(cand_row), max(max(r) for r in cur if r is not cand_row) if n_rows > 1 else 0)
            
            delta_e = cand_max - cur_val
            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                if random.random() < 0.5:
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
                    best = [r[:] for r in cur]
                    best_used = set(cur_used)
                    no_improve = 0
                else:
                    no_improve += 1
                
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        T = max(T_min, T * cool)
        
        if no_improve > 15000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set(best_used)
            cur_row_diffs = [_row_diffs_set(r) for r in cur]
            T = 20.0
            no_improve = 0
    
    if not _valid(best, n_rows, k):
        return _build_greedy(n_rows, k)
    return best

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with greedy
    rows = _build_greedy(n_rows, k)
    best_val = _max_entry(rows)
    
    # Try to improve using local search
    result = _local_search(rows, n_rows, k, deadline)
    
    # Verify validity
    if not _valid(result, n_rows, k):
        result = _build_greedy(n_rows, k)
    
    return result

# EVOLVE-BLOCK-END
