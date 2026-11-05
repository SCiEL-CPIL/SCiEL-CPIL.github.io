"""Difference triangle set (7,5): minimize the largest entry.
Global backtracking with MRV (Minimum Remaining Values) and Forward Checking.
Interleaves row construction to detect conflicts early."""
import time
import random
import math
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return the complete set of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used

def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _get_valid_candidates(row, used_diffs, max_val):
    """Get valid next values for a partial row."""
    if not row:
        return []
    last_val = row[-1]
    candidates = []
    # Optimization: Only check a limited range if max_val is large, but for small M we check all
    # Since M is ~105-112, range is small enough to iterate
    for v in range(last_val + 1, max_val + 1):
        # Check strict increase is satisfied by range
        new_diffs = {v - a for a in row}
        if not (new_diffs & used_diffs):
            candidates.append(v)
    return candidates

def _solve_interleaved(n_rows, k, max_val, deadline):
    """
    Solve using interleaved backtracking with MRV and Forward Checking.
    Returns list of rows or None.
    """
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    # We need to fill each row to length k+1
    # State: rows, used_diffs
    
    def is_complete():
        return all(len(r) == k + 1 for r in rows)
    
    def get_mrv_row():
        """Find the row with the fewest valid next candidates (MRV)."""
        min_count = float('inf')
        min_row_idx = -1
        min_cands = []
        
        for i in range(n_rows):
            if len(rows[i]) == k + 1:
                continue
            cands = _get_valid_candidates(rows[i], used_diffs, max_val)
            if len(cands) < min_count:
                min_count = len(cands)
                min_row_idx = i
                min_cands = cands
                if min_count == 0:
                    return -1, [] # Dead end
        return min_row_idx, min_cands

    def forward_check():
        """Check if any other incomplete row has 0 candidates."""
        for i in range(n_rows):
            if len(rows[i]) == k + 1:
                continue
            cands = _get_valid_candidates(rows[i], used_diffs, max_val)
            if len(cands) == 0:
                return False
        return True

    def backtrack():
        if time.time() > deadline:
            return None
        
        if is_complete():
            return [list(r) for r in rows]
        
        # Select row with MRV
        r_idx, candidates = get_mrv_row()
        
        if r_idx == -1:
            return None # Dead end
        
        # Sort candidates by "Least Constraining Value" heuristic?
        # For now, try in increasing order to keep scope small, 
        # or maybe randomize to escape symmetry? 
        # Given the tight bounds, increasing order is good for finding *a* solution quickly.
        # But to find *smaller* scopes, we rely on iterative deepening.
        
        for v in candidates:
            # Apply change
            rows[r_idx].append(v)
            new_diffs = {v - a for a in rows[r_idx][:-1]}
            old_used = set(used_diffs)
            used_diffs.update(new_diffs)
            
            # Forward Check
            if forward_check():
                # Recurse
                result = backtrack()
                if result is not None:
                    return result
            
            # Undo
            rows[r_idx].pop()
            used_diffs.difference_update(new_diffs)
            
        return None

    result = backtrack()
    return result

def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization. (Fallback)"""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                max_step = max(1, int((k + 1) * 2.5))
                found = False
                for _attempt in range(50):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows

def _local_search_improve(rows, k, deadline, rng=None):
    """Local search: try to reduce the maximum entry by modifying individual elements."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        for ri in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[ri]
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                if old_val < current_scope - 2:
                    continue
                
                low = row[pos - 1] + 1
                high = old_val - 1
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)
                
                if low > high:
                    continue
                
                n_candidates = min(10, high - low + 1)
                if n_candidates <= 0: continue
                candidates = rng.sample(range(low, high + 1), n_candidates)
                candidates.sort()
                
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                    if new_val >= old_val:
                        continue
                    
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(old_val - row[i])
                    for i in range(pos + 1, len(row)):
                        old_diffs.add(row[i] - old_val)
                    
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for i in range(pos + 1, len(row)):
                        new_diffs.add(row[i] - new_val)
                    
                    conflict = new_diffs & (used - old_diffs)
                    if conflict:
                        continue
                    
                    row[pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    improved = True
                    break
        
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_rows = None
    best_scope = float('inf')
    
    # Theoretical lower bound for m(7,5) is 105.
    # We iterate max_val from 105 upwards.
    # Interleaved backtracking is powerful, so we try small bounds first.
    
    # We will use a randomized restart strategy if the first attempt on a bound fails quickly
    # or if we hit the time limit.
    
    for max_val in range(105, 115):
        if time.time() > deadline - 10:
            break
            
        # Try a few randomizations of the search order?
        # The MRV selection is deterministic based on candidate count.
        # To break symmetry, we can shuffle the rows initially?
        # Actually, rows are symmetric in the problem definition.
        # We can shuffle the order in which we *try* candidates in MRV.
        
        # Let's run the solver.
        # To handle potential exponential blowup, we set a sub-deadline for each max_val attempt
        # and if it fails, we move to next max_val or try again with different random seed if we allow it.
        # For now, deterministic MRV + increasing candidates is a good start.
        
        # We'll allow 15 seconds per bound attempt initially, scaling down if time is tight.
        attempt_deadline = min(deadline - 5, time.time() + 20)
        
        rows = _solve_interleaved(n_rows, k, max_val, attempt_deadline)
        
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
            
            # If we found a solution, try to improve it with local search
            if time.time() < deadline - 10:
                improved = _local_search_improve(rows, k, deadline - 5)
                imp_scope = _scope(improved)
                if imp_scope < best_scope:
                    best_scope = imp_scope
                    best_rows = improved
                    
            # If we found a solution at the lower bound, we can stop
            if best_scope == 105:
                break
        else:
            # If it failed, it might be because the search space was too large or no solution exists.
            # Continue to next max_val.
            pass

    # If no solution found by interleaved backtracking, fallback to greedy + local search
    if best_rows is None:
        for seed_off in range(10):
            if time.time() > deadline - 5:
                break
            rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=seed_off)
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
        
        if best_rows is not None and time.time() < deadline - 5:
            improved = _local_search_improve(best_rows, k, deadline - 2)
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    return best_rows

# EVOLVE-BLOCK-END
