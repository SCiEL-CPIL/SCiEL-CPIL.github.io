"""Difference triangle set construction via backtracking search."""
import time
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry."""
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Each row has k+1 entries, so C(k+1,2) differences per row
    # Total differences needed: n_rows * C(k+1, 2)
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # The maximum possible difference is bounded by the largest entry
    # We need at least total_diffs distinct differences, so largest entry >= something
    # But we'll search for the optimal
    
    best_solution = None
    best_max_entry = float('inf')
    
    # Try different upper bounds for the largest entry
    # Lower bound: we need total_diffs distinct positive integers from 1..M
    # So M >= total_diffs
    # But also each row of k+1 elements has max element M, with C(k+1,2) diffs
    # The smallest possible max for one row is at least k (if consecutive)
    # But with the distinctness constraint across rows, it's larger
    
    # Simple lower bound: largest entry >= total_diffs (since diffs are 1..M and distinct)
    # Actually diffs can be up to M, and we need total_diffs of them, all distinct
    # So M >= total_diffs is not quite right; we need M such that there are at least
    # total_diffs distinct values in [1, M], so M >= total_diffs
    # But actually the max difference in a row with max entry M is at most M
    # So we need M >= total_diffs is a lower bound
    
    lower_bound = max(total_diffs, k)  # at least k for the row structure
    
    # Upper bound: greedy gives us some value, use that as initial upper bound
    greedy_sol = _greedy_construct(n_rows, k)
    if greedy_sol:
        best_solution = greedy_sol
        best_max_entry = max(max(row) for row in greedy_sol)
    
    # Try to improve from lower_bound up to best_max_entry
    # Use backtracking search for each candidate max
    for max_entry in range(lower_bound, best_max_entry + 1):
        if time.time() > deadline:
            break
        
        result = _backtracking_search(n_rows, k, max_entry)
        if result is not None:
            best_solution = result
            best_max_entry = max_entry
            break  # Found a solution with this max, can't do better
    
    if best_solution is None:
        # Fallback: return greedy
        return _greedy_construct(n_rows, k)
    
    return best_solution


def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes smallest next mark with unused differences."""
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


def _backtracking_search(n_rows, k, max_entry):
    """Backtracking search to find a valid configuration with all entries <= max_entry."""
    used = set()
    rows = []
    
    def backtrack(row_idx, current_row, pos_idx):
        """Try to complete row row_idx, which currently has current_row entries."""
        if time.time() > deadline:
            return None
        
        # Check if we've completed all rows
        if row_idx == n_rows:
            return rows[:]
        
        # If we're building a row and it's complete
        if pos_idx == k + 1:
            # Move to next row
            rows.append(current_row[:])
            result = backtrack(row_idx + 1, [0], 0)
            if result is not None:
                return result
            rows.pop()
            return None
        
        # Try candidate positions for the next entry
        # The next entry must be > current_row[-1] and <= max_entry
        # Also, the differences it creates must not conflict with used
        last = current_row[-1]
        
        # Generate candidates in order of "cost" - prefer those that use smaller differences
        # since small differences are more constrained
        candidates = []
        for x in range(last + 1, max_entry + 1):
            new_diffs = {x - a for a in current_row}
            # Check if any diff is already used
            if new_diffs & used:
                continue
            # Score: sum of differences (prefer smaller)
            # Also prefer positions that don't block future options
            score = sum(new_diffs)
            candidates.append((score, x))
        
        candidates.sort()
        
        for _, x in candidates:
            new_diffs = {x - a for a in current_row}
            
            # Tentatively add
            new_used = used | new_diffs
            new_row = current_row + [x]
            
            # Check feasibility: can we complete this row with remaining positions?
            # We need (k+1 - len(new_row)) more entries, each > x, <= max_entry
            remaining_slots = k + 1 - len(new_row)
            if remaining_slots > 0:
                # The smallest possible next entry is x+1, largest is max_entry
                # We need remaining_slots distinct values in [x+1, max_entry]
                if max_entry - x < remaining_slots:
                    continue
            
            # Recurse
            old_used = used
            used.clear()
            used.update(new_used)
            
            result = backtrack(row_idx, new_row, pos_idx + 1)
            
            used.clear()
            used.update(old_used)
            
            if result is not None:
                return result
        
        return None
    
    # Start with first row, first entry is 0
    return backtrack(0, [0], 0)


# EVOLVE-BLOCK-END
