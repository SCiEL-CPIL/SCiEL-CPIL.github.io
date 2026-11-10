import sys
import time

def construct(n_rows, k):
    # Initial greedy solution to get an upper bound
    def greedy(n_rows, k):
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

    initial_rows = greedy(n_rows, k)
    best_scope = max(max(row) for row in initial_rows)
    
    # Backtracking search to improve
    # We want to find a set of rows with max element < best_scope
    # Symmetry breaking: rows must be generated in increasing order of their last element.
    
    rows = []
    used_diffs = set()
    prev_max = -1
    
    # Total differences needed per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # We will search for a solution with max element <= best_scope - 1
    # If found, update best_scope and continue searching for even smaller?
    # Given time limits, finding one better solution is a good start.
    # We can do a branch and bound.
    
    def backtrack(row_idx, current_row, current_max):
        nonlocal best_scope, rows, used_diffs, prev_max
        
        if row_idx == n_rows:
            # Found a valid solution
            return True
        
        # Heuristic: The next row's max must be > prev_max
        # Also, we want to minimize the final max, so we try to keep current_max small.
        
        if len(current_row) == 0:
             # Start a new row
             # The first element is 0
             current_row = [0]
             current_max = 0
             # We need to choose the rest of the elements
             # To break symmetry, the last element of this row must be > prev_max
             # Let's choose the elements one by one.
        
        # Actually, let's structure the backtrack to fill one row at a time.
        # State: rows so far, used_diffs, prev_max (max element of last completed row)
        
        # Let's redefine the state to be cleaner.
        pass

    # Let's implement a cleaner backtracking that builds rows one by one.
    
    def solve():
        nonlocal best_scope
        
        # We will try to find a solution with max element < current best_scope
        # We iterate over possible max scopes? No, that's inefficient.
        # Instead, we use the current best_scope as a hard upper bound and try to find a solution.
        # If we find one, we update best_scope to the max element of that solution.
        # Then we can try again with the new tighter bound?
        # Given 120s, we might only have time for one or two improvements.
        
        # Let's do a single DFS that tries to find *any* solution with max element < best_scope.
        # We order the search to prefer smaller elements.
        
        rows = []
        used_diffs = set()
        prev_max = -1
        
        def dfs(row_idx):
            nonlocal best_scope, rows, used_diffs, prev_max
            
            if row_idx == n_rows:
                # Solution found
                # The scope is the max element of the last row (since rows are increasing in max)
                scope = rows[-1][-1]
                if scope < best_scope:
                    best_scope = scope
                    # We could return True to stop, but we might want to find even better?
                    # For now, let's just return True to indicate a better solution was found.
                    # The caller can decide if to continue.
                    return True
            
            # Try to construct row `row_idx`
            # The row must start with 0.
            # The elements must be strictly increasing.
            # The differences must be disjoint from used_diffs.
            # The last element must be > prev_max.
            
            # We will build the row element by element.
            # To simplify, we can try all possible rows that satisfy the conditions?
            # That's too many. We need to generate candidates.
            
            # Let's use a helper to generate the next valid element for a partial row.
            # But backtracking within a row is also needed.
            
            # Let's define a recursive function for building a single row.
            def build_row(row_idx, current_row, current_max, used_diffs_snapshot):
                # current_row: list of elements so far
                # current_max: last element
                # used_diffs_snapshot: set of diffs used in previous rows + diffs in current_row so far
                
                if len(current_row) == k + 1:
                    # Row complete
                    # Check symmetry: current_max must be > prev_max
                    if current_max <= prev_max:
                        return False
                    
                    # Save state
                    rows.append(current_row)
                    new_prev_max = current_max
                    new_used_diffs = used_diffs_snapshot.union({j - i for i in range(len(current_row)) for j in range(i+1, len(current_row))})
                    
                    # Recurse to next row
                    if dfs(row_idx + 1):
                        return True
                    
                    # Backtrack
                    rows.pop()
                    return False
                
                # Determine next element
                # The next element must be > current_max
                # And its differences with all previous elements in current_row must be in used_diffs_snapshot? No, they must NOT be in used_diffs_snapshot.
                
                # We need to find the smallest x > current_max such that {x - a for a in current_row} is disjoint from used_diffs_snapshot.
                # But we also need to ensure that it's possible to complete the row with max element < best_scope.
                
                # Heuristic: Try x from current_max + 1 upwards.
                # Pruning: If x >= best_scope, we can't improve, so stop.
                # Actually, if we are looking for a solution with max < best_scope, then all elements must be < best_scope.
                
                for x in range(current_max + 1, best_scope):
                    new_diffs = {x - a for a in current_row}
                    if new_diffs & used_diffs_snapshot:
                        continue
                    
                    # Check if it's possible to complete the row
                    # Remaining elements: k + 1 - len(current_row) - 1
                    # Remaining diffs needed for this row:
                    # Total diffs in a row of size m is m*(m-1)/2.
                    # Diffs already in this row: len(current_row)*(len(current_row)-1)/2
                    # Diffs involving x: len(current_row)
                    # This is getting complicated to estimate "possibility".
                    
                    # Simple pruning:
                    # The number of unused diffs in the range [1, best_scope-1] must be enough.
                    # But we don't track that easily.
                    
                    # Let's just try it.
                    new_row = current_row + [x]
                    new_used = used_diffs_snapshot.union(new_diffs)
                    
                    if build_row(row_idx, new_row, x, new_used):
                        return True
                
                return False
            
            # Start building the row with [0]
            # Diffs for [0] is empty.
            return build_row(row_idx, [0], 0, used_diffs)
        
        # Start DFS
        # We need to pass the initial state correctly.
        # The `used_diffs` in the outer scope is the one from previous rows.
        # In `dfs`, we call `build_row` which uses `used_diffs` from the outer scope of `dfs` (which is the `used_diffs` of the current row_idx level).
        # Wait, the closure in Python is tricky. Let's pass `used_diffs` explicitly.
        
        def dfs2(row_idx, used_diffs_so_far, prev_max_so_far):
            nonlocal best_scope
            
            if row_idx == n_rows:
                # Solution found
                # The scope is the max element of the last row added
                # We need to track the rows added.
                # Let's keep a global list of rows for the current path.
                # This is getting messy. Let's restart the design.
                return
        
        # Let's use a class or careful closure.
        
        current_rows = []
        
        def dfs3(row_idx, used_diffs, prev_max):
            nonlocal best_scope, current_rows
            
            if row_idx == n_rows:
                # Success
                scope = current_rows[-1][-1]
                if scope < best_scope:
                    best_scope = scope
                    return True
                return False
            
            # Build row
            def build_row_partial(current_row, current_max, used_diffs_local):
                if len(current_row) == k + 1:
                    if current_max <= prev_max:
                        return False
                    
                    # Check if max element is less than best_scope
                    if current_max >= best_scope:
                        return False
                    
                    current_rows.append(current_row)
                    new_used = used_diffs_local.union({j - i for i in range(len(current_row)) for j in range(i+1, len(current_row))})
                    
                    if dfs3(row_idx + 1, new_used, current_max):
                        return True
                    
                    current_rows.pop()
                    return False
                
                # Try next elements
                # Optimization: The differences must be unique.
                # We can pre-calculate which differences are forbidden.
                
                # We try x from current_max + 1 to best_scope - 1
                # We can add a pruning: if the number of remaining positions is R, we need R*(R+1)/2 + ... diffs?
                
                for x in range(current_max + 1, best_scope):
                    new_diffs = {x - a for a in current_row}
                    if new_diffs & used_diffs_local:
                        continue
                    
                    # Heuristic: If we are close to filling the row, check if remaining diffs can fit.
                    # For now, just recurse.
                    
                    if build_row_partial(current_row + [x], x, used_diffs_local.union(new_diffs)):
                        return True
                
                return False
            
            return build_row_partial([0], 0, used_diffs)
        
        # Start search
        if dfs3(0, set(), -1):
            # We found a better solution. We could continue to search for even better,
            # but let's return the current best_rows?
            # The function `construct` should return the best rows found.
            # We need to store the best rows.
            pass
        
        # This implementation is flawed because it doesn't store the best rows globally.
        # Let's refactor to store the best solution found.
        
        best_rows_found = None
        
        def dfs4(row_idx, used_diffs, prev_max, current_rows_path):
            nonlocal best_scope, best_rows_found
            
            if row_idx == n_rows:
                scope = current_rows_path[-1][-1]
                if scope < best_scope:
                    best_scope = scope
                    best_rows_found = [list(r) for r in current_rows_path]
                    # We can return True to indicate improvement, but we might want to find more.
                    # Let's return True to stop this branch, but the outer loop can restart with tighter bound.
                    return True
                return False
            
            def build_row_partial(current_row, current_max, used_diffs_local):
                if len(current_row) == k + 1:
                    if current_max <= prev_max:
                        return False
                    if current_max >= best_scope:
                        return False
                    
                    new_rows = current_rows_path + [current_row]
                    new_used = used_diffs_local.union({j - i for i in range(len(current_row)) for j in range(i+1, len(current_row))})
                    
                    if dfs4(row_idx + 1, new_used, current_max, new_rows):
                        return True
                    return False
                
                for x in range(current_max + 1, best_scope):
                    new_diffs = {x - a for a in current_row}
                    if new_diffs & used_diffs_local:
                        continue
                    
                    if build_row_partial(current_row + [x], x, used_diffs_local.union(new_diffs)):
                        return True
                return False
            
            return build_row_partial([0], 0, used_diffs)
        
        # Run search
        # We might run out of time. We should add a time check.
        start_time = time.time()
        
        # We can try to improve the greedy solution iteratively.
        # First run: find any solution < greedy_scope
        if dfs4(0, set(), -1, []):
             # Found one. We can try to find an even better one?
             # Let's do a second pass with the new best_scope.
             if time.time() - start_time < 60:
                 # Reset best_rows_found to allow finding a better one
                 # But dfs4 updates best_scope and best_rows_found.
                 # We need to ensure we don't return the old one if a better one is found.
                 # The logic inside dfs4 updates best_rows_found.
                 # So we just run it again? No, dfs4 uses best_scope as a bound.
                 # If we call dfs4 again, it will search for < new_best_scope.
                 pass
        
        if best_rows_found:
            return best_rows_found
        else:
            return initial_rows

# This is getting complex and error-prone with closures. 
# A simpler, more robust approach for small k:
# Use `itertools` or simple recursion with clear state passing.
# Given the constraints, a simple backtracking with symmetry breaking and a time limit is best.

def construct(n_rows, k):
    import time
    
    def greedy(n_rows, k):
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

    initial_rows = greedy(n_rows, k)
    best_scope = max(max(row) for row in initial_rows)
    best_rows = initial_rows
    
    start_time = time.time()
    
    # Backtracking search
    # State: row_idx, used_diffs, prev_max
    # We build rows one by one.
    
    def dfs(row_idx, used_diffs, prev_max, current_rows):
        nonlocal best_scope, best_rows
        
        if time.time() - start_time > 110:
            return False
        
        if row_idx == n_rows:
            scope = current_rows[-1][-1]
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in current_rows]
                return True
            return False
        
        # Build current row
        # We need to choose k elements after 0.
        # We will do this recursively.
        
        def build_row(current_row, current_max, used_diffs_local):
            if time.time() - start_time > 110:
                return False
            
            if len(current_row) == k + 1:
                if current_max <= prev_max:
                    return False
                if current_max >= best_scope:
                    return False
                
                new_used = used_diffs_local.union({j - i for i in range(len(current_row)) for j in range(i+1, len(current_row))})
                return dfs(row_idx + 1, new_used, current_max, current_rows + [current_row])
            
            # Try next element
            # Pruning: The smallest possible next element is current_max + 1.
            # The largest possible is best_scope - 1.
            # We can also prune if the number of remaining unused differences is too small.
            
            for x in range(current_max + 1, best_scope):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs_local:
                    continue
                
                # Heuristic: Check if it's possible to complete the row.
                # Remaining elements: rem = k + 1 - len(current_row) - 1
                # We need rem more elements.
                # The differences involving these elements must be unique.
                
                if build_row(current_row + [x], x, used_diffs_local.union(new_diffs)):
                    return True
            return False
        
        return build_row([0], 0, used_diffs)
    
    # Start search
    dfs(0, set(), -1, [])
    
    return best_rows
