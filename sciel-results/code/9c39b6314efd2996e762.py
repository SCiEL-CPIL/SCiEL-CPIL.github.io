import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _generate_sidon_rows(max_val, k):
    """
    Generate all Sidon sets of size k+1 starting at 0, with max entry <= max_val.
    Returns a list of tuples (max_entry, row_tuple, diff_set).
    """
    rows = []
    # We use backtracking to generate the row
    # row[0] = 0
    # We need to choose k more numbers from 1 to max_val
    
    # To be efficient, we can prune if the number of remaining slots is not enough
    # or if the current differences already conflict within the row.
    
    def backtrack(current_row, current_diffs, start_val):
        if len(current_row) == k + 1:
            # Valid Sidon set
            # Check if all diffs are unique (should be by construction)
            rows.append((max(current_row), tuple(current_row), frozenset(current_diffs)))
            return
        
        # Heuristic: The smallest possible max for the remaining elements
        # If we need 'need' more elements, the smallest max is roughly 
        # current_last + need (if we pick consecutive, but they must be Sidon)
        # Actually, just iterate.
        
        # Pruning: If we can't fit the remaining elements in [start_val, max_val]
        # The minimum span for 'need' Sidon elements is not trivial, but 
        # a simple bound is: we need at least 'need' numbers, so 
        # max_val - start_val + 1 >= need is necessary.
        need = (k + 1) - len(current_row)
        if max_val - start_val + 1 < need:
            return
        
        for v in range(start_val, max_val + 1):
            # Check if v creates a difference conflict with existing row elements
            new_diffs = set()
            valid = True
            for a in current_row:
                d = v - a
                if d in current_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                # Check if new_diffs have internal conflicts?
                # No, Sidon property requires all pairwise diffs to be distinct.
                # We added diffs with existing elements.
                # We don't have new pairs among new elements yet, only one new element.
                # So we just need to ensure d not in current_diffs.
                
                backtrack(current_row + [v], current_diffs | new_diffs, v + 1)

    backtrack([0], set(), 1)
    return rows

def _find_disjoint_rows(candidate_rows, n_rows):
    """
    Find n_rows rows from candidate_rows such that their difference sets are disjoint.
    Returns the list of rows if found, else None.
    candidate_rows is a list of (max_val, row_tuple, diff_set).
    """
    # Sort by max_val to try to find small solutions first?
    # Or just iterate.
    # To speed up, we can order candidates by "popularity" of their diffs?
    # Simpler: Just backtrack.
    
    # We want to minimize the max of the selected rows.
    # The candidate_rows are already generated up to a specific max_val limit.
    # We will check if a set of n_rows exists.
    
    # Optimization: If n_rows is small (7), we can use a recursive search.
    # To prune, we can check if the remaining candidates can provide enough disjoint diffs.
    
    used_diffs = set()
    selected = []
    
    def backtrack(idx, used_diffs, selected):
        if len(selected) == n_rows:
            return selected[:]
        
        # Pruning: If we have selected 'len(selected)' rows, we need 'n_rows - len(selected)' more.
        # If there are not enough candidates left, return None.
        if len(candidate_rows) - idx < n_rows - len(selected):
            return None
            
        for i in range(idx, len(candidate_rows)):
            max_val, row, diffs = candidate_rows[i]
            
            # Check disjointness
            if not (diffs & used_diffs):
                # Accept
                res = backtrack(i + 1, used_diffs | diffs, selected + [row])
                if res is not None:
                    return res
            # If conflict, skip
            
        return None

    return backtrack(0, set(), [])

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # The task is for n_rows=7, k=5 (6 entries).
    # Best known is 112.
    # We will iterate M from a lower bound up to the best known.
    
    # Lower bound: Counting bound.
    # Total diffs = n_rows * (k+1)*k/2 = 7 * 15 = 105.
    # So M must be at least 105.
    # Let's start from 105.
    
    # However, generating all Sidon rows up to M for each M is expensive.
    # Better: Generate Sidon rows incrementally or for a large M, then filter.
    
    # Let's try M = 112 first (known best). If we find a solution, we can try to improve.
    # If we don't find one (unlikely for 112), we increase M.
    
    # To be safe and fast, we'll generate rows up to M_max = 120.
    # Then we will try to find a solution with max <= M for M in [105, 120].
    
    M_max = 120
    all_rows = _generate_sidon_rows(M_max, k)
    
    # Sort all_rows by max_val
    all_rows.sort(key=lambda x: x[0])
    
    # For each possible max limit M, filter rows with max <= M and try to find n_rows disjoint.
    # We can optimize by just checking if a solution exists for M=105, 106, etc.
    
    best_solution = None
    best_max = float('inf')
    
    # We can binary search or linear search. Linear is fine for small range.
    # But generating rows is the expensive part.
    # So we generate once for M_max.
    
    # We need to find the minimum M such that a disjoint set of n_rows exists among rows with max <= M.
    
    # To speed up the disjoint search, we can use a heuristic:
    # Try rows with smaller max first?
    # The backtrack function above does this implicitly if we sort by max.
    
    # However, the search space for disjoint sets can be large.
    # We can add a time check.
    
    # Let's try M from 105 to 115.
    for M in range(105, M_max + 1):
        if time.time() > deadline:
            break
            
        # Filter rows with max <= M
        # Instead of filtering every time, we can use the sorted list.
        # Find index of first row with max > M.
        # Since all_rows is sorted by max_val, we can just take a prefix.
        
        # Binary search for the cutoff index
        lo, hi = 0, len(all_rows)
        while lo < hi:
            mid = (lo + hi) // 2
            if all_rows[mid][0] <= M:
                lo = mid + 1
            else:
                hi = mid
        count = lo
        
        if count < n_rows:
            continue
            
        candidate_rows = all_rows[:count]
        
        # Try to find disjoint rows
        # To make it faster, we can randomize the order of candidates slightly?
        # No, deterministic is better for reproducibility, but might get stuck.
        # Let's try the deterministic backtrack first.
        
        sol = _find_disjoint_rows(candidate_rows, n_rows)
        
        if sol is not None:
            best_solution = sol
            best_max = M
            break # Found the minimum M for this M range
            
    if best_solution is None:
        # If we didn't find one in the range, or if we timed out,
        # fall back to a greedy/local search result or the best found so far.
        # For robustness, if no solution found, return a valid greedy set.
        # But 112 is known to exist.
        
        # Let's try a quick local search on a greedy start to ensure we return *something* valid.
        # But the prompt asks for a stronger algorithm.
        # If the backtracking failed, it might be too slow or the pruning was bad.
        
        # Let's implement a fallback: Greedy construction.
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
        best_solution = rows
        best_max = max(max(r) for r in rows)

    # Convert tuples to lists
    final_rows = [list(r) for r in best_solution]
    
    # Verify validity
    used_diffs = set()
    valid = True
    for r in final_rows:
        if len(r) != k + 1 or r[0] != 0:
            valid = False
            break
        for i in range(len(r)):
            if i > 0 and r[i] <= r[i-1]:
                valid = False
                break
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in used_diffs:
                    valid = False
                    break
                used_diffs.add(d)
        if not valid:
            break
            
    if not valid:
        # Fallback to greedy if something went wrong
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
        final_rows = rows

    return final_rows

# EVOLVE-BLOCK-END
