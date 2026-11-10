import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115 # 5s buffer
    
    # The task is specific: n_rows=7, k=5 (6 elements per row).
    # But we should handle general n_rows, k if possible, or at least the specific case.
    # The prompt says "evaluator passes this task's n_rows and k".
    # The known best for (7,5) is 112.
    
    # Lower bound for max value m:
    # Number of differences per row = k*(k+1)/2 = 5*6/2 = 15.
    # Total differences = n_rows * 15 = 7 * 15 = 105.
    # Differences must be distinct integers in [1, m].
    # So m >= 105.
    
    min_m = n_rows * (k * (k + 1) // 2)
    
    # We will try to find a solution with max element <= M
    # Iterate M from min_m upwards.
    # To avoid taking too long if min_m is too low and unsolvable, we can limit the range.
    # For (7,5), range is 105 to 112.
    
    # Bitmask approach:
    # Max difference is M. We need a bitmask of size M+1.
    # Python integers have arbitrary precision, so this is fine.
    
    def solve_for_m(M):
        # We need to construct n_rows rows.
        # Each row has k+1 elements: 0 = a_0 < a_1 < ... < a_k <= M.
        # We will build rows one by one.
        # To break symmetry, we can require that the rows are sorted lexicographically?
        # Or simply that the last element of row i is <= last element of row i+1?
        # Actually, sorting rows by their last element is a good symmetry break.
        
        # State:
        # row_idx: 0..n_rows-1
        # pos: 1..k (0 is already placed)
        # current_row: list of ints
        # used_mask: int
        
        # We use a recursive function with memoization?
        # State space might be too large for memoization without careful hashing.
        # Simple backtracking with strong pruning should work for small N.
        
        # Pruning:
        # 1. If current max element in current_row > M, fail.
        # 2. Check if new differences conflict with used_mask.
        # 3. Forward checking: Can we still complete the current row?
        #    - We have placed `pos` elements. We need `k - pos` more.
        #    - The next element must be > current_row[-1].
        #    - The differences created must be new.
        
        # To improve speed, we can pre-calculate possible next steps?
        # Or just iterate candidates.
        
        # Let's add a timeout check inside the recursion.
        
        # We'll use a list to store the solution rows.
        solution = []
        
        def backtrack(row_idx, pos, current_row, used_mask):
            if time.time() - start_time > time_limit:
                return False, None
            
            if row_idx == n_rows:
                # All rows constructed
                return True, solution[:]
            
            if pos == k + 1:
                # Current row complete
                # Symmetry breaking: The last element of this row must be <= last element of next row?
                # Since we don't know next row yet, we can enforce that rows are sorted by their last element.
                # So, if solution is not empty, current_row[-1] >= solution[-1][-1]
                if solution and current_row[-1] < solution[-1][-1]:
                    return False, None
                solution.append(current_row[:])
                return backtrack(row_idx + 1, 1, [0], used_mask)
            
            # We need to choose current_row[pos]
            # It must be > current_row[pos-1]
            # And <= M
            # And differences {current_row[pos] - a for a in current_row[:pos]} must not be in used_mask
            
            last_val = current_row[pos-1]
            # Optimization: We can estimate the minimum possible value for the remaining elements
            # to see if we can stay within M?
            # The remaining elements must be strictly increasing.
            # min_next = last_val + 1
            # min_next_next = last_val + 2 ...
            # So the last element will be at least last_val + (k - pos + 1).
            # If last_val + (k - pos + 1) > M, we can't fit the rest?
            # Actually, the last element is current_row[k].
            # If we are at pos, we need k-pos+1 more elements (indices pos to k).
            # The smallest they can be is last_val+1, last_val+2, ..., last_val+(k-pos+1).
            # So current_row[k] >= last_val + (k - pos + 1).
            if last_val + (k - pos + 1) > M:
                return False, None
            
            # Candidates for current_row[pos]
            # We iterate from last_val + 1 to M
            # But we can limit the upper bound.
            # The max value we can pick for current_row[pos] is M - (k - pos)
            # because we need k-pos more elements after it.
            max_candidate = M - (k - pos)
            
            # Also, we should try to keep the row "compact" to leave room for differences?
            # Or maybe spread out?
            # Let's try iterating from low to high first (greedy-like) to find valid rows quickly.
            # If that fails, we might need to backtrack to try higher values earlier.
            
            for val in range(last_val + 1, max_candidate + 1):
                # Check differences
                # new_diffs = {val - a for a in current_row[:pos]}
                # new_mask = 0
                # for a in current_row[:pos]:
                #     d = val - a
                #     if d > M: break # shouldn't happen
                #     new_mask |= (1 << d)
                
                # Efficient mask calculation
                new_mask = 0
                for a in current_row[:pos]:
                    d = val - a
                    new_mask |= (1 << d)
                
                if new_mask & used_mask:
                    continue
                
                # Forward check: Are there enough unused differences for the rest of the row?
                # This is hard to check exactly.
                # Simple check: Is the new value valid?
                
                # Recurse
                current_row.append(val)
                success, sol = backtrack(row_idx, pos + 1, current_row, used_mask | new_mask)
                if success:
                    return True, sol
                current_row.pop()
            
            return False, None
        
        # Start backtracking
        # Initial row: [0]
        # We need to handle the symmetry of the first row too?
        # The first row can be anything.
        # But to reduce search space, we can fix the first row?
        # No, the first row is part of the optimization.
        
        success, sol = backtrack(0, 1, [0], 0)
        return sol

    # Try increasing M
    # We know the answer is 112.
    # We can start from min_m and go up.
    # To save time, we can start from a known lower bound or just iterate.
    # For (7,5), min_m=105.
    
    # Heuristic: Start from a slightly higher bound if min_m is likely unsolvable?
    # 105, 106, 107, 108, 109, 110, 111, 112.
    # 8 attempts. If each takes < 10s, we are fine.
    
    # To speed up the first few attempts (which are likely to fail or be slow),
    # we can use a randomized or heuristic order for candidates?
    # The current backtracking tries low values first.
    
    # Let's add a timeout per M attempt?
    # If an M takes too long, we might skip it? No, we need the exact min.
    # But if 105 is unsolvable, the search space might be huge.
    # We can limit the time per M. If it exceeds, we assume it's unsolvable?
    # Risky. But for this specific problem, 105-111 are likely unsolvable or very hard.
    # 112 is the known solution.
    
    # Let's try to be smart:
    # If we find a solution for M, return it.
    
    for M in range(min_m, min_m + 20): # Upper bound heuristic
        if time.time() - start_time > time_limit - 5:
            break
        sol = solve_for_m(M)
        if sol:
            return sol
            
    # Fallback: Greedy
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
    return rows
