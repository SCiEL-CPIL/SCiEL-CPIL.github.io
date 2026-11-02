import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of length k+1.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # The known optimal scope is 112 (bound) to 134 (known best).
    # We will search for the smallest valid scope.
    # Counting bound: n_rows * k * (k+1) / 2 differences must fit in [1, M].
    # Actually, differences are distinct across all rows.
    # Total differences = n_rows * k * (k+1) / 2.
    # For n=4, k=8: 4 * 8 * 9 / 2 = 144 differences.
    # Max difference is M. So M must be at least 144? 
    # Wait, the differences are $a_j - a_i$. The largest difference is $M - 0 = M$.
    # The set of differences is a subset of $\{1, \dots, M\}$.
    # So we need at least 144 distinct differences. Thus $M \ge 144$?
    # Let's re-read the counting bound.
    # "simple counting bound is 112".
    # Why 112?
    # Maybe the differences don't have to cover all integers up to M?
    # Yes, they just have to be distinct.
    # The largest difference is $M$. The smallest is 1.
    # We need 144 distinct values from $\{1, \dots, M\}$.
    # So $M \ge 144$.
    # But the prompt says "simple counting bound is 112".
    # Let me check the math.
    # Number of differences per row: k(k+1)/2.
    # Total: n * k(k+1)/2.
    # For n=4, k=8: 4 * 36 = 144.
    # If M=112, we only have 112 possible difference values.
    # We need 144 distinct values. This is impossible.
    # Is the prompt implying something else?
    # "all differences between two entries of the same row are distinct over all rows"
    # Yes, global distinctness.
    # Maybe k is different? "Choose 4 rows of 8 increasing integers".
    # So k=8? "k+1 integers". So length 9?
    # "list of k+1 integers that starts at 0".
    # If k=8, length is 9.
    # Differences per row: 9*8/2 = 36.
    # Total: 4 * 36 = 144.
    # If M=112, we have 112 slots. 144 > 112. Contradiction.
    
    # Let's re-read carefully: "Choose 4 rows of 8 increasing integers starting at 0".
    # This phrasing is ambiguous. Does "8 increasing integers" mean the row has 8 elements?
    # Or does it mean k=8, so 8+1=9 elements?
    # The interface says: "Each row is a list of k + 1 integers".
    # The task says: "Choose 4 rows of 8 increasing integers...".
    # Usually "n integers" means length n.
    # If length is 8, then k=7.
    # Differences per row: 8*7/2 = 28.
    # Total: 4 * 28 = 112.
    # Ah! If k=7 (length 8), the counting bound is exactly 112.
    # So k=7.
    
    # Let's assume the evaluator passes k=7.
    # If the evaluator passes k=8, the bound would be 144.
    # The prompt says "The best known set has largest entry 134... counting bound is 112".
    # This strongly implies the instance is n=4, length=8 (k=7).
    # I will write the code to handle any n_rows, k passed by the evaluator.
    # But I will optimize for the case where the bound is tight.
    
    # Let's determine the lower bound for M.
    total_diffs = n_rows * k * (k + 1) // 2
    min_scope = total_diffs
    
    # Search for M from min_scope upwards
    # We know 134 is a known good value.
    # We can start from min_scope and go up.
    
    # To speed up, we can use a backtracking solver with a fixed M.
    # We will try M = min_scope, min_scope+1, ... until we find a solution.
    # Since 134 is known, and 112 is the bound, the gap is small.
    
    def solve_with_scope(M):
        used_diffs = set()
        rows = []
        
        def backtrack_row(row_idx, current_row):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                if row_idx + 1 == n_rows:
                    return True
                return backtrack_row(row_idx + 1, [0])
            
            # Determine next element x
            # Heuristic: try small x first to keep scope tight
            # But we must ensure x <= M
            start_val = current_row[-1] + 1 if current_row else 1
            
            for x in range(start_val, M + 1):
                # Check differences
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Pruning: Can we fit the remaining elements?
                # Remaining elements count: (k + 1) - len(current_row) - 1
                # This is complex to prune effectively without more info.
                # Simple pruning: if x is too large, we might not be able to add more?
                # No, we can always add larger numbers up to M.
                
                # Try adding x
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack_row(row_idx, current_row):
                    return True
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        return backtrack_row(0, [0])

    # Try scopes
    # We can jump a bit if we know 134 works.
    # But let's start from the bound to find the true minimum.
    # For n=4, k=7, min_scope=112.
    # We'll try 112, 113, ...
    
    # Optimization: If n_rows and k are such that the search space is huge,
    # we might need a better heuristic.
    # For now, simple DFS with M increasing is robust for small instances.
    
    # To handle the case where the evaluator passes different n, k,
    # we should be careful.
    
    # Let's add a time check in the outer loop too.
    current_M = min_scope
    # We can set an upper bound on the search to avoid infinite loops if no solution found (shouldn't happen)
    max_search_M = min_scope + 50  # 134 is 22 above 112. 50 is safe.
    
    while current_M <= max_search_M:
        if time.time() > deadline:
            break
            
        # Reset state for each M
        # The solver above modifies 'rows' and 'used_diffs' via closure.
        # We need to ensure they are fresh for each M attempt.
        # Let's refactor solve_with_scope to be self-contained or reset properly.
        
        # Refactored solver
        def solve_fixed_M(M):
            used_diffs = set()
            rows = []
            
            def bt(row_idx, current_row):
                if time.time() > deadline:
                    return False
                
                if len(current_row) == k + 1:
                    rows.append(current_row)
                    if row_idx + 1 == n_rows:
                        return True
                    return bt(row_idx + 1, [0])
                
                start_val = current_row[-1] + 1 if current_row else 1
                
                # Heuristic: Try values that allow more flexibility?
                # Small values are generally better for minimizing max.
                for x in range(start_val, M + 1):
                    new_diffs = set()
                    valid = True
                    for a in current_row:
                        d = x - a
                        if d in used_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    current_row.append(x)
                    used_diffs.update(new_diffs)
                    
                    if bt(row_idx, current_row):
                        return True
                    
                    current_row.pop()
                    used_diffs.difference_update(new_diffs)
                
                return False
            
            if bt(0, [0]):
                return rows
            return None

        result = solve_fixed_M(current_M)
        if result:
            return result
        
        current_M += 1
        
        # If we are taking too long on one M, we might skip to next?
        # No, we must find the minimum.
        # But if it's too slow, we might miss the deadline.
        # For 4x8 (k=7), DFS should be fast enough for M~112-134.
        
    # If we run out of time or scopes, return a greedy solution as fallback
    # Greedy:
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
