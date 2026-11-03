import time
import sys
import math
from collections import defaultdict

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0

def construct(n_rows, k):
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 110.0
    
    # n_rows = 7, k = 5 (so 6 elements per row)
    # We want to find 7 rows of 6 integers starting at 0, strictly increasing.
    # All pairwise differences within a row must be unique across all rows.
    # Minimize the largest element (scope).
    
    # Counting bound:
    # Each row has C(k+1, 2) differences. For k=5, that's 15 differences.
    # Total differences = 7 * 15 = 105.
    # Differences are in range [1, max_scope].
    # So max_scope >= 105.
    # Best known is 112.
    
    # Strategy: Backtracking search.
    # State:
    # - rows: list of current rows (lists of ints)
    # - used_diffs: set of all differences used so far
    # - current_scope: max element in last row (since we build rows with increasing max? 
    #   Actually, we don't have to enforce row max order, but for pruning we can assume 
    #   we are trying to keep the global max low. 
    #   However, a better heuristic is to just minimize the max element of the *last* row 
    #   if we sort rows by their max element? No, that's restrictive.
    #   Let's just track the global max scope so far.
    
    # Heuristic: 
    # 1. Order rows by some heuristic? Or just build them one by one.
    # 2. When building a row, choose elements one by one.
    # 3. Pruning: If the current element exceeds the best known scope, prune.
    # 4. Additionally, we can estimate the minimum possible scope for the remaining rows.
    #    Remaining rows need `remaining_rows * 15` distinct differences.
    #    The smallest possible differences are 1, 2, ..., N.
    #    So if we have used `u` differences, the next available ones are at least `u+1`? No, they can be any unused number.
    #    But the maximum difference in a row is bounded by its max element.
    #    If we want to fit `r` more rows, and we have used `used_diffs`, the new differences must be distinct from `used_diffs` and each other.
    #    This is a complex constraint.
    
    # Simpler pruning:
    # - Keep a global `best_scope` found so far (initialize with a greedy solution).
    # - In the backtracking tree, if the current row's last element exceeds `best_scope`, prune.
    # - Also, if the number of available differences <= `best_scope` is less than `total_needed`, prune.
    
    # Let's first run a fast greedy to get an initial upper bound.
    
    def fast_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for a in row:
                    if (x - a) in used:
                        valid = False
                        break
                if valid:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            rows.append(row)
        return rows

    init_rows = fast_greedy()
    best_scope = max(a for r in init_rows for a in r[1:])
    best_rows = [list(r) for r in init_rows]
    
    # Now, try to improve with backtracking.
    # We will try to find a solution with scope < best_scope.
    # We can iterate target_scope from best_scope - 1 downwards? 
    # Or just use best_scope as a hard limit and try to find any solution, 
    # then reduce the limit if found.
    
    # To be efficient, we should try to find a solution with scope = target.
    # Let's start with target = best_scope - 1 and go down to the counting bound (105).
    # If we find a solution with scope S, update best_scope = S and try S-1.
    
    # Backtracking function
    # rows: list of completed rows
    # used_diffs: set of used differences
    # row_idx: index of the row currently being built
    # current_row: list of elements in the current row
    # target_scope: the maximum allowed value for any element
    
    def backtrack(row_idx, current_row, used_diffs, target_scope):
        if time.time() > _DEADLINE:
            return None
        
        if row_idx == n_rows:
            return [list(r) for r in [current_row] + rows] # Wait, rows is passed by reference? 
            # Better to pass a list of rows and modify it, or return the constructed rows.
            # Let's restructure: 
            # backtrack(row_idx, rows, used_diffs, target_scope)
            # where rows is a list of lists.
            # When row_idx == n_rows, return rows.
            
    # Let's rewrite properly.
    
    def solve(target_scope):
        # used_diffs is a set of integers
        # rows is a list of lists
        # We build rows one by one.
        
        # Optimization: Represent used_diffs as a bitset if possible? 
        # Max scope ~112, so differences up to 112. 
        # Python int can be used as a bitset. 
        # bit i set means difference i is used.
        
        def build_row(used_bits, target_scope):
            # Returns a list of 6 integers starting with 0, strictly increasing, 
            # such that all differences are not in used_bits and are <= target_scope.
            # Also, the last element must be <= target_scope.
            
            # Heuristic: Try to keep elements as small as possible? 
            # Or randomize?
            # For now, deterministic greedy within the row, with backtracking if needed.
            # Since k=5, the row has 6 elements. 
            # We can try to generate the row by picking the next element from the smallest available.
            
            row = [0]
            used_local = 0 # Bitset of differences used in this row
            
            def dfs(pos, last_val):
                if time.time() > _DEADLINE:
                    return None
                if pos == k + 1:
                    return row[:]
                
                # Try next values from last_val + 1 to target_scope
                # To improve search, we might want to try values that are "spread out" 
                # or use a heuristic. 
                # For now, let's try from last_val + 1 upwards.
                # This tends to produce "compact" rows, which might be good for minimizing scope.
                
                for x in range(last_val + 1, target_scope + 1):
                    # Check if x is a valid next element
                    # 1. Differences x - a for a in row must not be in used_bits
                    # 2. Differences x - a for a in row must not be in used_local
                    # 3. x - a must be > 0 (guaranteed by x > last_val >= a)
                    
                    valid = True
                    new_local = used_local
                    for a in row:
                        d = x - a
                        if d > target_scope: # Difference cannot exceed scope? 
                            # Actually, difference is bounded by the max element in the row, 
                            # which is <= target_scope. So d <= target_scope is always true.
                            pass
                        if (used_bits >> d) & 1:
                            valid = False
                            break
                        if (new_local >> d) & 1:
                            valid = False
                            break
                        new_local |= (1 << d)
                    
                    if not valid:
                        continue
                    
                    row.append(x)
                    result = dfs(pos + 1, x)
                    if result is not None:
                        return result
                    row.pop()
                
                return None
            
            return dfs(1, 0)
        
        rows = []
        used_bits = 0
        
        def backtrack_rows(row_idx):
            if time.time() > _DEADLINE:
                return False
            
            if row_idx == n_rows:
                return True
            
            # Generate a valid row for this index
            # We can try multiple rows? 
            # The order of rows doesn't matter for validity, but might matter for search efficiency.
            # Let's just try to build one row. If it fails, we need to backtrack?
            # No, if build_row fails, it means no valid row exists given the current used_bits.
            # But wait, build_row is deterministic/greedy. It might fail even if a valid row exists.
            # So we need backtracking *within* build_row as well? 
            # The dfs in build_row already does backtracking for the elements of the row.
            # But what if the *choice* of which row to build next matters?
            # Since rows are symmetric, we can just build them in order.
            # However, if build_row fails, it means no row can be built with the current used_bits.
            # This implies the current set of previous rows is a dead end.
            # But we didn't backtrack on the *previous* rows' specific choices, only their existence.
            # Actually, the previous rows were also built via dfs, which explores alternatives.
            # But once a row is committed, we don't revisit it.
            # This is a "sequential" backtracking. 
            # If row 2 fails, we should backtrack to row 1 and try a different row 1.
            
            # So we need a more integrated backtracking.
            
            # Let's restructure: 
            # The state is (row_idx, current_row_elements, used_bits).
            # We choose the next element for the current row.
            # If we complete a row, we move to the next row.
            
            # This is essentially the same as the dfs in build_row, but nested.
            
            # Let's write a single recursive function that handles both row selection and element selection.
            
            pass
        
        # Let's implement a unified backtracker.
        
        def unified_backtrack(row_idx, current_row, used_bits):
            if time.time() > _DEADLINE:
                return None
            
            if row_idx == n_rows:
                # All rows built
                if len(current_row) == 0:
                    return rows[:]
                else:
                    # This shouldn't happen if we reset current_row
                    return None
            
            if len(current_row) == k + 1:
                # Row is complete
                rows.append(current_row)
                # Reset current_row for next row
                result = unified_backtrack(row_idx + 1, [0], used_bits)
                if result is not None:
                    return result
                rows.pop()
                return None
            
            # Otherwise, we are building the current row
            last_val = current_row[-1]
            
            # Try next values
            # Heuristic: Try values that are likely to leave room for other rows?
            # For now, try from last_val + 1 to target_scope.
            # To improve, we can randomize the order or use a heuristic.
            # Let's try a mix: first few smallest, then some random.
            
            # Optimization: If the number of available differences is too low, prune.
            # Total differences needed: (n_rows - row_idx) * 15 - (len(current_row)*(len(current_row)+1)//2 - 1) ?
            # No, just check if we can possibly fit the remaining differences.
            
            # Let's just iterate x.
            # To speed up, we can skip x if x - last_val is too small? No.
            
            # Randomization: Shuffle the range? 
            # For now, deterministic ascending.
            
            for x in range(last_val + 1, target_scope + 1):
                valid = True
                new_used_bits = used_bits
                for a in current_row:
                    d = x - a
                    if (new_used_bits >> d) & 1:
                        valid = False
                        break
                    new_used_bits |= (1 << d)
                
                if not valid:
                    continue
                
                current_row.append(x)
                result = unified_backtrack(row_idx, current_row, new_used_bits)
                if result is not None:
                    return result
                current_row.pop()
            
            return None
        
        rows = []
        # Start with row 0, element 0
        return unified_backtrack(0, [0], 0)
    
    # Try to find a solution with scope < best_scope
    # We can try target_scope from best_scope - 1 down to 105.
    
    found = False
    for target in range(best_scope - 1, 104, -1):
        if time.time() > _DEADLINE:
            break
        
        # We need to restart the search for each target.
        # To make it faster, we can use the previous solution as a hint?
        # For now, just run the backtracker.
        
        # The unified_backtrack above uses a global `rows` list which is messy.
        # Let's clean it up.
        
        def solve_target(target_scope):
            # Local state
            rows = []
            
            def backtrack(row_idx, current_row, used_bits):
                if time.time() > _DEADLINE:
                    return None
                
                if row_idx == n_rows:
                    if len(current_row) == 0:
                        return [list(r) for r in rows]
                    return None
                
                if len(current_row) == k + 1:
                    rows.append(list(current_row))
                    result = backtrack(row_idx + 1, [0], used_bits)
                    if result is not None:
                        return result
                    rows.pop()
                    return None
                
                last_val = current_row[-1]
                
                # Heuristic: Try to spread elements?
                # For now, ascending order.
                
                for x in range(last_val + 1, target_scope + 1):
                    valid = True
                    new_used_bits = used_bits
                    for a in current_row:
                        d = x - a
                        if (new_used_bits >> d) & 1:
                            valid = False
                            break
                        new_used_bits |= (1 << d)
                    
                    if not valid:
                        continue
                    
                    current_row.append(x)
                    result = backtrack(row_idx, current_row, new_used_bits)
                    if result is not None:
                        return result
                    current_row.pop()
                
                return None
            
            return backtrack(0, [0], 0)
        
        solution = solve_target(target)
        if solution is not None:
            best_scope = target
            best_rows = solution
            found = True
            # We can break here if we want the first solution found, 
            # but we want the *minimum* scope. 
            # Since we are iterating downwards, the first one found is the best for this loop.
            # But we should continue to see if we can go lower?
            # The loop continues to target-1.
            # So we don't break, we just update best.
            # However, if we find a solution at 111, we should try 110.
            # The loop does that.
            
    return best_rows

# Test
if __name__ == "__main__":
    rows = construct(7, 5)
    for r in rows:
        print(r)
    print("Scope:", max(a for r in rows for a in r[1:]))
