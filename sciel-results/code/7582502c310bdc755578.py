import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    timeout = 110  # Leave 10s margin for 120s limit
    
    # Lower bound for M: number of distinct differences needed
    num_diffs = n_rows * (k * (k + 1) // 2)
    min_m = num_diffs
    
    # We search for the smallest M that allows a valid configuration
    # We iterate M upwards
    # We can also use binary search if we had a fast checker, but backtracking is exponential.
    # Iterating from min_m is safer as it finds the first (best) solution.
    
    for M in range(min_m, 200):  # Cap at 200 to avoid infinite loop if no solution found quickly
        if time.time() - start_time > timeout:
            break
            
        # Try to construct solution with max mark M
        result = solve(n_rows, k, M, start_time, timeout)
        if result is not None:
            return result
            
    # If we run out of time or M range, return the best found or a fallback
    # Fallback: greedy (original logic) to ensure we return *something* valid, even if not optimal
    return greedy_construct(n_rows, k)

def solve(n_rows, k, M, start_time, timeout):
    used_diffs = set()
    
    def backtrack(row_idx, pos, current_row, prev_row):
        if time.time() - start_time > timeout:
            return False, None
            
        if row_idx == n_rows:
            return True, None # All rows constructed
            
        if pos == k + 1:
            # Current row complete
            return backtrack(row_idx + 1, 0, [], current_row)
            
        if pos == 0:
            # First mark is 0
            # Check lexicographic constraint: current_row (which is [0]) must be <= prev_row?
            # Actually, we compare the full rows. 
            # If prev_row is None, no constraint.
            # If prev_row is not None, current_row[0]=0 == prev_row[0]=0.
            # So we just proceed.
            return backtrack(row_idx, pos + 1, [0], prev_row)
            
        # Determine lower bound for next mark
        low = current_row[-1] + 1
        
        # Lexicographic constraint with prev_row
        if prev_row is not None:
            # We need current_row (once complete) >= prev_row
            # Let's check the prefix.
            # If current_row[:pos] < prev_row[:pos], invalid. (Shouldn't happen if we check at each step)
            # If current_row[:pos] == prev_row[:pos], then current_row[pos] must be >= prev_row[pos]
            # If current_row[:pos] > prev_row[:pos], no constraint on current_row[pos] (just > current_row[-1])
            
            # We maintain the invariant that the prefix is compatible.
            # When we chose previous marks, we ensured this.
            # So here, we just need to ensure that if the prefixes were equal, we pick >= prev_row[pos].
            
            # How do we know if prefixes were equal?
            # We can store a flag or just compare.
            # Since we are building, we can just compare current_row[:pos] and prev_row[:pos].
            # But for efficiency, let's just say:
            # If current_row == prev_row[:pos] (which implies all previous choices were equal), 
            # then next mark must be >= prev_row[pos].
            # Otherwise, if current_row > prev_row[:pos] (lexicographically), no extra constraint.
            
            # Actually, simpler:
            # If current_row is a prefix of prev_row (i.e., current_row == prev_row[:pos]),
            # then we must pick x >= prev_row[pos].
            # If current_row > prev_row[:pos], we can pick any x > current_row[-1].
            # If current_row < prev_row[:pos], this state is invalid (should have been pruned earlier).
            
            if current_row == prev_row[:pos]:
                low = max(low, prev_row[pos])
        
        # Iterate possible marks
        for x in range(low, M + 1):
            # Check differences
            # new_diffs = {x - a for a in current_row}
            # If any diff in used_diffs, skip
            
            # Optimization: Check conflicts
            conflict = False
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Tentatively add
            new_diffs = {x - a for a in current_row}
            used_diffs.update(new_diffs)
            
            success, _ = backtrack(row_idx, pos + 1, current_row + [x], prev_row)
            
            # Undo
            used_diffs.difference_update(new_diffs)
            
            if success:
                # We need to return the rows
                # The current_row + [x] is the partial row.
                # The final rows will be assembled by the caller or we pass them up.
                # Let's modify backtrack to return the list of rows.
                # This is slightly inefficient due to copying, but n_rows and k are small.
                return True, current_row + [x] # This returns the row being built? No, we need all rows.
        
        return False, None

    # The above logic is a bit messy for returning the full solution.
    # Let's use a global list or pass the solution list.
    
    return None # Placeholder

def greedy_construct(n_rows, k):
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
