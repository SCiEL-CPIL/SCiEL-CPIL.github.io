"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses a backtracking approach with pruning:
    - We build rows one by one.
    - For each row, we try to find a set of k+1 marks (starting with 0)
      whose differences don't conflict with previously used differences.
    - We use a greedy heuristic with backtracking to minimize the max entry.
    """
    
    # Total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # The counting bound: we need at least total_diffs_needed distinct differences,
    # and the max difference m must satisfy m*(m+1)//2 >= total_diffs_needed
    # So m >= smallest integer such that m*(m+1)//2 >= total_diffs_needed
    
    def min_scope(total):
        # Find smallest m such that m*(m+1)//2 >= total
        m = 0
        while m * (m + 1) // 2 < total:
            m += 1
        return m
    
    lower_bound = min_scope(total_diffs_needed)
    
    # Try to find a solution with max entry = lower_bound, then lower_bound+1, etc.
    # For each scope, try to pack the rows.
    
    def try_construct(scope):
        """Try to construct n_rows rows with all entries <= scope."""
        # All possible differences from 1 to scope
        # We need to partition a subset of {1..scope} into n_rows sets, each of size diffs_per_row
        # where each set corresponds to the differences of a valid ruler (k+1 marks starting at 0)
        
        # First, let's think about it differently:
        # A row is determined by its k marks (excluding 0). The differences are all pairwise differences.
        # We need to find n_rows such that all their differences are disjoint.
        
        # Approach: backtracking over rows. For each row, find a valid set of k marks
        # (from 1..scope) such that the differences don't conflict with used_diffs.
        
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, used_diffs, rows):
            if row_idx == n_rows:
                return True
            
            # Try to find a valid row
            # The row has marks: 0 = m0 < m1 < m2 < ... < mk <= scope
            # Differences: all mj - mi for i < j
            
            # We'll build the row greedily with backtracking on the marks
            row = [0]
            
            def build_row(row, next_mark_idx, used_diffs):
                if next_mark_idx == k:
                    # Row is complete
                    rows.append(row[:])
                    if backtrack(row_idx + 1, used_diffs, rows):
                        return True
                    rows.pop()
                    return False
                
                # Try each possible next mark
                start = row[-1] + 1
                for mark in range(start, scope + 1):
                    # Check if adding this mark creates any new difference that's already used
                    new_diffs = {mark - a for a in row}
                    if new_diffs & used_diffs:
                        continue
                    
                    # Also check: the new mark must be <= scope
                    row.append(mark)
                    new_used = used_diffs | new_diffs
                    
                    if build_row(row, next_mark_idx + 1, new_used):
                        return True
                    
                    row.pop()
                
                return False
            
            if build_row(row, 1, used_diffs):
                return True
            
            return False
        
        return backtrack(0, used_diffs, rows)
    
    # Try scopes from lower_bound upward
    # But the search space can be huge, so we need a smarter approach.
    
    # Alternative: Use a more efficient backtracking with ordering.
    # Let's use the fact that we can order the rows by their largest mark.
    
    # Actually, let's try a different approach: 
    # 1. Generate all possible "rulers" (sets of k marks from 1..scope with 0) 
    # 2. For each ruler, compute its difference set
    # 3. Find n_rows rulers with disjoint difference sets
    
    # But generating all rulers is expensive. Let's use backtracking with pruning.
    
    def solve(scope):
        used_diffs = set()
        rows = []
        
        def bt(row_idx):
            if row_idx == n_rows:
                return True
            
            # Try to place row row_idx
            # We need k marks: m1 < m2 < ... < mk, all in [1, scope]
            # such that all differences are not in used_diffs
            
            # To make this tractable, we'll use a recursive search with pruning
            # We build the row mark by mark
            
            def place_marks(marks, next_idx):
                if next_idx == k:
                    # Complete row
                    rows.append(marks[:])
                    if bt(row_idx + 1):
                        return True
                    rows.pop()
                    return False
                
                start = (marks[-1] + 1) if marks else 1
                for m in range(start, scope + 1):
                    # Check differences with all existing marks in this row
                    new_diffs = {m - a for a in marks}
                    if new_diffs & used_diffs:
                        continue
                    
                    # Pruning: check if the remaining differences can still fit
                    # (This is a simple check; more sophisticated pruning could be added)
                    
                    marks.append(m)
                    if place_marks(marks, next_idx + 1):
                        return True
                    marks.pop()
                
                return False
            
            return place_marks([], 0)
        
        if bt(0):
            return rows
        return None
    
    # Try increasing scopes
    for scope in range(lower_bound, lower_bound + 50):
        result = solve(scope)
        if result is not None:
            return result
    
    # If we couldn't find a solution with a reasonable scope, fall back to greedy
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
