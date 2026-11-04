import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0
    
    # Theoretical lower bound
    # n_rows * (k*(k-1)/2) differences
    num_diffs = n_rows * (k * (k - 1) // 2)
    lower_bound = num_diffs
    
    # We know 112 is a good upper bound.
    # We try from lower_bound up to 112 (or a bit higher if needed)
    # If we find a solution, we return it.
    
    best_rows = None
    best_scope = float('inf')
    
    # Fallback greedy
    def deterministic_greedy():
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

    best_rows = deterministic_greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # Backtracking solver
    def solve_for_scope(M):
        used_diffs = set()
        rows = []
        
        # Symmetry breaking: rows must be sorted by their max element
        # prev_max tracks the max element of the last completed row
        # Initially -infinity, but since rows start at 0, and increase, 
        # the first row can have any max.
        # However, we can enforce that the max of row i >= max of row i-1.
        
        def backtrack(row_idx, current_row, last_val, prev_max):
            nonlocal used_diffs, rows, best_rows, best_scope
            
            if time.time() - start_time > deadline:
                return False, "timeout"
                
            if row_idx == n_rows:
                # All rows constructed
                return True, None
                
            if len(current_row) == k + 1:
                # Row complete
                row_max = current_row[-1]
                # Check symmetry constraint
                if row_max < prev_max:
                    return False, None
                
                # Check if we can still complete remaining rows
                # Remaining rows: n_rows - row_idx - 1
                # Each needs k+1 elements, max >= row_max
                # This is a weak check, but we can estimate.
                
                rows.append(current_row)
                # Save new diffs
                new_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        new_diffs.add(current_row[j] - current_row[i])
                used_diffs.update(new_diffs)
                
                ok, reason = backtrack(row_idx + 1, [0], 0, row_max)
                
                if ok:
                    return True, None
                
                # Backtrack
                rows.pop()
                used_diffs.difference_update(new_diffs)
                return False, None
            
            # Choose next value x
            # x must be > last_val and <= M
            # Also, the final max of this row must be >= prev_max
            # If current_row is not full, the max will be at least the last element added.
            # So if last_val < prev_max, we MUST add something >= prev_max eventually.
            # Actually, if len(current_row) < k+1, the max is not fixed yet.
            # But the max of the completed row must be >= prev_max.
            
            start_x = last_val + 1
            # If we are at the last position (k-th index, 0-based), x must be >= prev_max
            # If we are not at the last position, x can be small, but subsequent values must push max up.
            
            candidates = []
            for x in range(start_x, M + 1):
                # Check if adding x creates conflicts
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                # Also, new_diffs must be <= M? No, diffs are just values.
                # But the elements are <= M.
                
                candidates.append(x)
                
            if not candidates:
                return False, None
            
            # Heuristic: Shuffle candidates? Or sort by some metric?
            # Randomization helps escape local minima in search tree.
            random.shuffle(candidates)
            
            # Try candidates
            for x in candidates:
                # If we are adding the last element, ensure it satisfies max constraint
                if len(current_row) == k:
                    if x < prev_max:
                        continue
                        
                new_diffs = {x - a for a in current_row}
                # Double check
                if new_diffs & used_diffs:
                    continue
                    
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                ok, reason = backtrack(row_idx, current_row, x, prev_max)
                if ok:
                    return True, None
                    
                current_row.pop()
                used_diffs.difference_update(new_diffs)
                
                if reason == "timeout":
                    return False, "timeout"
                    
            return False, None
        
        # Start backtracking
        # First row prev_max = -1 (any max is ok)
        # Actually, to break symmetry fully, we can fix the first row?
        # No, just start with prev_max = -1.
        ok, reason = backtrack(0, [0], 0, -1)
        
        if ok:
            return rows
        else:
            return None

    # Try scopes from lower bound up to best_scope
    # We know 112 is a solution.
    # We should try 105, 106, ... up to min(best_scope, 112)
    
    # To ensure we don't waste time on impossible small M,
    # we can check if a solution exists for M.
    
    # We will try in increasing order of M.
    # If we find one, we return it immediately (it's better than best_scope).
    
    max_search_scope = min(best_scope, 112)
    
    # Optimization: If best_scope is already 112, we only try 105..111.
    # If best_scope is 118, we try 105..117.
    
    for M in range(lower_bound, max_search_scope + 1):
        if time.time() - start_time > deadline:
            break
            
        # Run solver
        # Add a time limit per M? 
        # If we spend too long on M=105, we might miss M=112.
        # But 105 is the lower bound. If it's not 105, it's likely not 106 either?
        # Not necessarily.
        
        # Let's allocate a small time budget per M, or just let the global deadline handle it.
        # Given the small space, backtracking should be fast or fail fast.
        
        res = solve_for_scope(M)
        if res is not None:
            best_rows = res
            best_scope = M
            # We found a solution for M. Since we are iterating upwards, this is the best for this M.
            # But maybe a smaller M was skipped? No, we iterate from lower_bound.
            # So the first M that works is the optimal (or near optimal) scope.
            # We can break if we found a solution for M == lower_bound.
            if M == lower_bound:
                break
            # Should we continue to see if a smaller one exists? No, we started from lower_bound.
            # So this is the smallest M we found.
            # We can break because we are looking for the minimum scope.
            # Wait, if we found a solution for M=108, is it possible there is one for M=107?
            # We iterated 105, 106, 107... so if we found 108, it means 105, 106, 107 failed.
            # So 108 is the best we found.
            break
            
    return best_rows
