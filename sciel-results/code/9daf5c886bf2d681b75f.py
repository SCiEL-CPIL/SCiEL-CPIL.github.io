"""Difference triangle set (7,5) solver using branch-and-bound on a compatibility graph."""

import numpy as np
from itertools import combinations
from collections import defaultdict
import time


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    # For the specific task: n_rows=7, k=5 (6 entries per row)
    # We'll use a branch-and-bound approach with greedy construction and local search.
    
    # First, try to find a solution with a small scope using a smart greedy + backtracking approach.
    
    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid_row(row, k):
        """Check if a row is valid: starts at 0, strictly increasing, length k+1."""
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        return True
    
    # Strategy: Use a combination of greedy construction with backtracking
    # and a local improvement phase.
    
    # For small instances, we can use a more systematic approach.
    # The key insight: we want to minimize the maximum element across all rows.
    
    # Let's try a recursive backtracking approach with pruning.
    # For each row, we build it incrementally, ensuring no difference conflicts.
    
    best_solution = None
    best_scope = float('inf')
    
    # Upper bound from simple greedy
    def greedy_construct(n_rows, k, max_scope):
        """Greedy construction with a maximum scope constraint."""
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            success = False
            for _attempt in range(k):
                found = False
                for x in range(current_max + 1, max_scope + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        current_max = x
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        
        return rows
    
    # Try greedy with increasing scope
    for scope in range(k * (k + 1) // 2, 200):
        result = greedy_construct(n_rows, k, scope)
        if result is not None:
            actual_scope = max(max(row) for row in result)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_solution = result
            break
    
    # If greedy didn't find a good solution, try a more sophisticated approach
    if best_solution is None:
        # Fallback: just use greedy without scope limit
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        best_solution = rows
        best_scope = max(max(row) for row in best_solution)
    
    # Now try to improve the solution using local search
    # We'll try swapping elements within rows or trying different row constructions
    
    def try_improve(rows, k, n_rows, max_iterations=5000):
        """Try to improve the solution by reconstructing rows with a tighter scope."""
        current_scope = max(max(row) for row in rows)
        best = [list(r) for r in rows]
        best_scope = current_scope
        
        used_diffs = set()
        for row in best:
            used_diffs |= get_diffs(row)
        
        for iteration in range(max_iterations):
            # Try to reconstruct one row at a time with a smaller max element
            improved = False
            for idx in range(n_rows):
                # Remove this row's differences
                row_diffs = get_diffs(best[idx])
                remaining_diffs = used_diffs - row_diffs
                
                # Try to build a new row with a smaller max element
                for new_max in range(best[idx][-1] - 1, 0, -1):
                    new_row = _construct_row_with_limit(remaining_diffs, k, new_max)
                    if new_row is not None:
                        new_diffs = get_diffs(new_row)
                        # Check no conflicts with other rows
                        if not (new_diffs & (used_diffs - row_diffs)):
                            # Accept the improvement
                            used_diffs = (used_diffs - row_diffs) | new_diffs
                            best[idx] = new_row
                            new_scope = max(max(r) for r in best)
                            if new_scope < best_scope:
                                best_scope = new_scope
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best, best_scope
    
    def _construct_row_with_limit(forbidden_diffs, k, max_val):
        """Construct a row of k+1 elements starting at 0, max <= max_val,
        with no differences in forbidden_diffs."""
        row = [0]
        
        def backtrack(row):
            if len(row) == k + 1:
                return row
            last = row[-1]
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden_diffs:
                    continue
                row.append(x)
                result = backtrack(row)
                if result is not None:
                    return result
                row.pop()
            return None
        
        return backtrack(row)
    
    # Run local improvement
    best_solution, best_scope = try_improve(best_solution, k, n_rows)
    
    return best_solution
