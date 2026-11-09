import time
import sys
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    # The problem:
    # We need n_rows rows, each of length k+1.
    # Row r is a sequence a_r[0], a_r[1], ..., a_r[k] with a_r[0]=0 and strictly increasing.
    # All differences a_r[j] - a_r[i] for i < j must be distinct across all rows and all pairs.
    # Total differences = n_rows * (k+1)*k/2.
    # Let D = n_rows * (k+1)*k/2.
    # We want to minimize the maximum value in any row (the scope).
    
    # Strategy: Backtracking search on the differences.
    # We can view the construction as filling a matrix of differences.
    # However, the differences in a row are not independent.
    # a_r[j] = sum_{t=i+1}^j (a_r[t] - a_r[t-1])
    # Let d_r[i] = a_r[i] - a_r[i-1] for i=1..k. These are the "steps".
    # Then a_r[j] = sum_{t=1}^j d_r[t].
    # The differences are a_r[j] - a_r[i] = sum_{t=i+1}^j d_r[t].
    
    # This is still complex to propagate.
    
    # Alternative Strategy:
    # Since k is small (5 in this task, so k+1=6), the number of differences per row is 15.
    # Total differences for 7 rows is 105.
    # The scope M is roughly sqrt(2 * 105) ~ 14? No, the differences are distinct integers.
    # The largest difference in a row is at most M.
    # The set of all differences is a subset of {1, ..., M}.
    # We need to partition the set of used differences into n_rows sets of size k(k+1)/2,
    # such that each set can be realized as the differences of an increasing sequence of length k+1 starting at 0.
    
    # This is a "Difference Basis" problem.
    
    # Let's try a randomized local search on the set of differences?
    # Or a backtracking search on the rows themselves, but with a better heuristic.
    
    # The current program uses greedy initialization and then local search on elements.
    # The directive suggests a CSP approach.
    
    # Let's define the variables as the elements of the rows.
    # Variable: a[r][j] for r in 0..n_rows-1, j in 0..k.
    # a[r][0] = 0.
    # Constraints:
    # 1. a[r][j] < a[r][j+1]
    # 2. All differences a[r][j] - a[r][i] (i<j) are distinct.
    
    # We want to minimize max(a[r][k]).
    
    # We can perform a backtracking search that tries to build the rows one by one,
    # or interleaved.
    # Given the small size (7 rows, 6 elements), a backtracking search with strong pruning might work.
    
    # Pruning:
    # - Maintain a set of used differences.
    # - When adding a new element, check if its differences with previous elements in the same row are already used.
    # - Also, we should bound the search by a current best scope.
    
    # To find a good scope quickly, we can use a binary search on M or start from a greedy estimate and decrease.
    
    # Let's first get a baseline using the greedy method from the current program.
    
    def greedy_init():
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

    best_rows = greedy_init()
    best_scope = max(max(r) for r in best_rows)
    
    # Now, try to improve using backtracking search for a smaller scope.
    # We will try scopes from best_scope - 1 downwards.
    # For each scope M, we check if a solution exists.
    
    # To speed up the backtracking, we can order the rows by their "difficulty" or just fix an order.
    # Actually, the rows are symmetric. We can fix the first row to be the "greedy" one?
    # No, that might be suboptimal.
    
    # Let's use a randomized backtracking search.
    # We will try to fill the rows one by one.
    # For each row, we try to find a valid sequence of length k+1 starting at 0 with max <= M,
    # such that its differences don't conflict with previously used differences.
    
    # This decomposes the problem into:
    # 1. Find a sequence for row 0 with max <= M.
    # 2. Find a sequence for row 1 with max <= M, disjoint diffs from row 0.
    # ...
    # 7. Find a sequence for row 6 with max <= M, disjoint diffs from rows 0-5.
    
    # This is a valid approach because the constraints between rows are only via the disjointness of differences.
    # The constraints within a row are handled by the sequence generation.
    
    # How to generate a valid sequence for a row given a set of forbidden differences?
    # We can use backtracking to build the row element by element.
    
    def find_row(forbidden_diffs, M, random_seed=None):
        """
        Finds a row of length k+1 starting at 0, strictly increasing, max <= M,
        such that all internal differences are not in forbidden_diffs.
        Returns the row or None.
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        used_in_row = set()
        row = [0]
        
        def backtrack(pos, current_max):
            if pos == k + 1:
                return row[:]
            
            # Determine range for next element
            # Must be > row[-1] and <= M
            # Also, the difference with 0 is the element itself.
            
            # Heuristic: try smaller values first to keep the row compact?
            # Or randomize to explore different structures?
            # Let's try a mix.
            
            start_val = row[-1] + 1
            end_val = M
            
            # Optimization: If we are at the last position, we just need to pick a value.
            # If not, we need to ensure there's enough room for the remaining k+1-pos elements.
            # Min value for next is row[-1] + 1.
            # Max value is M.
            
            # Let's iterate from start_val to end_val.
            # To avoid bias, we can shuffle the order if we fail?
            # For now, let's try increasing order.
            
            for val in range(start_val, end_val + 1):
                # Check differences with all previous elements in the row
                valid = True
                new_diffs = set()
                for i in range(pos):
                    d = val - row[i]
                    if d in forbidden_diffs or d in used_in_row:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check if this choice allows completion?
                # Simple check: The maximum possible difference for the last element
                # if we pick the smallest possible next steps.
                # This is complex. Let's just rely on backtracking.
                
                row.append(val)
                used_in_row.update(new_diffs)
                
                result = backtrack(pos + 1, val)
                if result is not None:
                    return result
                
                row.pop()
                used_in_row -= new_diffs
                
            return None
        
        return backtrack(1, 0)

    # The function find_row is deterministic if we iterate in order.
    # To find different solutions, we can randomize the order of values tried.
    
    def find_row_random(forbidden_diffs, M, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        used_in_row = set()
        row = [0]
        
        def backtrack(pos):
            if pos == k + 1:
                return row[:]
            
            start_val = row[-1] + 1
            end_val = M
            
            candidates = list(range(start_val, end_val + 1))
            random.shuffle(candidates)
            
            for val in candidates:
                valid = True
                new_diffs = set()
                for i in range(pos):
                    d = val - row[i]
                    if d in forbidden_diffs or d in used_in_row:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                row.append(val)
                used_in_row.update(new_diffs)
                
                result = backtrack(pos + 1)
                if result is not None:
                    return result
                
                row.pop()
                used_in_row -= new_diffs
                
            return None
        
        return backtrack(1)

    # Now, the main search loop.
    # We try to reduce the scope.
    # We start from best_scope - 1 and go down.
    
    # To make the search more effective, we can try multiple random restarts for each M.
    
    current_scope = best_scope
    
    while current_scope > 0 and time.time() < deadline:
        # Try to find a solution with scope = current_scope
        found = False
        attempts = 0
        max_attempts = 50 # Limit attempts per scope to avoid getting stuck
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            if attempts > 1:
                random.seed(int(time.time() * 1000) + attempts)
            
            used_diffs = set()
            current_rows = []
            success = True
            
            for r in range(n_rows):
                # Try to find a row
                row = find_row_random(used_diffs, current_scope)
                if row is None:
                    success = False
                    break
                current_rows.append(row)
                # Add differences to used_diffs
                for i in range(k + 1):
                    for j in range(i + 1, k + 1):
                        used_diffs.add(row[j] - row[i])
            
            if success:
                found = True
                best_rows = current_rows
                best_scope = current_scope
                break
        
        if found:
            # We found a solution with this scope. Try to go lower.
            current_scope -= 1
        else:
            # Failed to find a solution with this scope.
            # It's possible a solution exists but our heuristic failed to find it.
            # However, for small instances, if we try enough random restarts,
            # we are likely to find it if it exists.
            # If we fail, we assume the lower bound is higher.
            # We can break here, or try a few more times?
            # Let's break to be safe with time.
            break

    return best_rows

