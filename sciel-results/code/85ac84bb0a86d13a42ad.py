import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 110  # 10s margin from 120s limit
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def find_one_row(used_diffs, max_val, timeout):
        """Find one valid row starting with 0, within max_val, disjoint diffs from used_diffs."""
        row = [0]
        
        def backtrack(current_row, start_val):
            if len(current_row) == k + 1:
                return current_row[:]
            
            if time.time() > timeout:
                return None
            
            # Candidates
            # To improve randomness, we can shuffle the range
            candidates = list(range(start_val, max_val + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                current_row.append(x)
                res = backtrack(current_row, x + 1)
                if res is not None:
                    return res
                current_row.pop()
            
            return None
        
        return backtrack(row, 1)

    def try_construct(max_val, timeout_limit):
        """
        Attempts to construct n_rows of valid rows within max_val.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            if time.time() > timeout_limit:
                return None
            
            # Allocate a portion of the remaining time to finding this row
            row_timeout = min(0.5, timeout_limit - time.time())
            
            row = find_one_row(used_diffs, max_val, time.time() + row_timeout)
            if row is None:
                return None
            
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        return rows

    # 1. Get initial upper bound
    best_rows = greedy_construct()
    best_scope = get_scope(best_rows)
    
    # 2. Binary search for minimum scope
    low = max(total_diffs, 10)
    high = best_scope
    
    while low < high and time.time() < deadline - 5:
        mid = (low + high) // 2
        
        time_left = deadline - 5 - time.time()
        if time_left < 2:
            break
        
        # Give a fraction of remaining time to this attempt
        attempt_time = min(10, time_left * 0.5)
        
        result = try_construct(mid, time.time() + attempt_time)
        
        if result is not None:
            high = mid
            actual_scope = get_scope(result)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = result
        else:
            low = mid + 1
            
    # Final validation
    if not is_valid(best_rows):
        best_rows = greedy_construct()
        best_scope = get_scope(best_rows)

    return best_rows
