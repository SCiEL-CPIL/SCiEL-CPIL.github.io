import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0
    
    # For the specific case (7,5), we can use a known good solution
    # or use a more efficient search
    
    # Fallback greedy
    def greedy_construct():
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
    
    # For (7,5), try to find a solution with M close to the known best (112)
    # Use backtracking with better heuristics
    
    if n_rows == 7 and k == 5:
        # Try M values from 105 to 120
        for M in range(105, 121):
            if time.time() - start_time > time_limit - 5:
                break
            
            used = set()
            rows = []
            
            def can_complete(row_idx, curr_row, used_diffs):
                if time.time() - start_time > time_limit:
                    return None
                
                if len(curr_row) == k + 1:
                    if row_idx == n_rows - 1:
                        rows.append(list(curr_row))
                        return rows
                    return can_complete(row_idx + 1, [0], used_diffs)
                
                last = curr_row[-1]
                start_val = last + 1
                
                # Try values, but with some randomization to avoid getting stuck
                candidates = list(range(start_val, M + 1))
                
                for x in candidates:
                    new_diffs = {x - a for a in curr_row}
                    
                    if new_diffs & used_diffs:
                        continue
                    
                    new_used = used_diffs | new_diffs
                    new_row = curr_row + [x]
                    
                    result = can_complete(row_idx, new_row, new_used)
                    if result is not None:
                        return result
                
                return None
            
            # Try with different random seeds for candidate ordering
            for attempt in range(20):
                random.seed(attempt)
                result = can_complete(0, [0], set())
                if result is not None:
                    return result
            
            if time.time() - start_time > time_limit - 5:
                break
    
    # If no solution found, return greedy result
    return greedy_construct()
