import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 135.0  # 15s safety margin for 150s limit

    # The number of differences in one row is k * (k+1) / 2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # Get initial solution via greedy
    best_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(best_rows)

    # Lower bound for M
    M_lower = max(total_diffs_needed, k * (k + 1) // 2 + k, 20)

    # Try to find a solution with smaller M using randomized search
    for M in range(M_lower, best_M):
        if time.time() > deadline - 10:
            break
        
        # Try to find a packing with max entry <= M
        rows = _solve_direct(n_rows, k, M, deadline)
        
        if rows is not None and _is_valid(rows, k):
            # Found a valid solution
            new_M = _max_entry(rows)
            if new_M < best_M:
                best_rows = rows
                best_M = new_M
                # Since we're iterating M upward, if we found a valid solution,
                # it might not have max entry exactly M, but <= M
                # Continue trying smaller M? No, we're going up.
                # Actually, we should break since we found the minimal M that works?
                # Not necessarily - the search might not be exhaustive.
                # But for now, let's continue to see if we can do better with more search time
                # Actually, let's stop since we found a solution for this M
                break
    
    return best_rows

def _solve_direct(n_rows, k, M, deadline):
    """
    Direct backtracking solver with constraint propagation and randomized restarts.
    Builds rows one by one, choosing values that don't conflict with used differences.
    """
    random.seed(int(time.time() * 1000) % (2**31))
    
    # Precompute all possible valid single rows up to M for reference
    # But we won't enumerate them all; we'll generate on the fly
    
    # Strategy: 
    # 1. Use a priority queue of "row slots" to fill
    # 2. For each slot, try to find a valid row using backtracking
    # 3. Use constraint propagation: when a row is placed, update available differences
    
    best_result = None
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break
            
        rows = _backtrack_with_propagation(n_rows, k, M, deadline, restart)
        
        if rows is not None:
            best_result = rows
            break
    
    return best_result

def _backtrack_with_propagation(n_rows, k, M, deadline, restart_id):
    """
    Backtracking solver that builds rows with constraint propagation.
    Uses randomized ordering for restart diversity.
    """
    used_diffs = set()
    rows = []
    
    # Determine row construction order
    # For restart diversity, sometimes shuffle the order
    if restart_id % 2 == 0:
        row_order = list(range(n_rows))
    else:
        row_order = list(range(n_rows))
        random.shuffle(row_order)
    
    # For each row in order, try to construct it
    for row_idx in row_order:
        if time.time() > deadline - 15:
            return None
            
        row = _construct_single_row(k, M, used_diffs, deadline)
        if row is None:
            return None
        
        rows.append(row)
        # Update used differences
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                used_diffs.add(d)
    
    # Reorder rows to original index
    ordered_rows = [None] * n_rows
    for i, row_idx in enumerate(row_order):
        ordered_rows[row_idx] = rows[i]
    
    return ordered_rows

def _construct_single_row(k, M, used_diffs, deadline):
    """
    Construct a single valid row of k+1 integers starting at 0,
    with max entry <= M, such that all its differences are not in used_diffs.
    """
    # Backtracking search for a valid row
    # State: current partial row, current differences
    
    def backtrack(start, chosen, current_diffs):
        if time.time() > deadline - 15:
            return None
            
        if len(chosen) == k:
            return [0] + chosen[:]
        
        # Pruning: if we can't fit k more values in the remaining range
        remaining_slots = k - len(chosen)
        available_range = M - start + 1
        if available_range < remaining_slots:
            return None
        
        # Try values in randomized order for diversity
        candidates = list(range(start, M + 1))
        random.shuffle(candidates)
        
        for v in candidates:
            # Check if v creates any new difference conflict
            ok = True
            new_diffs = set()
            for a in chosen:
                d = v - a
                if d in current_diffs or d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            
            if not ok:
                continue
            
            # Check that new_diffs are all distinct from each other
            if len(new_diffs) != len(chosen):
                continue
            
            # Additional pruning: check if the remaining differences can fit
            # The smallest possible next value is v+1, so min new diff is 1
            # But we need to ensure we can complete the row
            
            chosen.append(v)
            current_diffs |= new_diffs
            
            result = backtrack(v + 1, chosen, current_diffs)
            if result is not None:
                return result
            
            chosen.pop()
            current_diffs -= new_diffs
        
        return None
    
    return backtrack(1, [], set())

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False
            
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
