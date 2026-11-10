import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    def greedy():
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

    def backtrack(limit, time_limit):
        used = 0
        rows = []
        
        def dfs(row_idx, current_row, last_val):
            if time.time() > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row[:])
                if dfs(row_idx + 1, [0], 0):
                    return True
                rows.pop()
                return False
            
            start = last_val + 1
            remaining_slots = k - len(current_row)
            
            for x in range(start, limit + 1):
                if x + remaining_slots > limit:
                    break
                
                new_mask = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if used & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                used |= new_mask
                current_row.append(x)
                
                if dfs(row_idx, current_row, x):
                    return True
                
                current_row.pop()
                used &= ~new_mask
            
            return False
        
        if dfs(0, [0], 0):
            return rows
        return None

    rows_greedy = greedy()
    best_rows = rows_greedy
    best_M = max(max(r) for r in best_rows)
    
    # Lower bound estimation
    # M(M+1) >= n_rows * k * (k+1)
    lb = 1
    while lb * (lb + 1) < n_rows * k * (k + 1):
        lb += 1
    # The problem states counting bound is 105 for this specific case.
    # We can use a slightly higher bound to be safe, or just use the calculated one.
    # Let's use max(lb, 10)
    
    limit = best_M - 1
    
    while limit > lb and time.time() < deadline:
        # Allow 5 seconds per attempt
        attempt_deadline = min(deadline - 1.0, time.time() + 5.0)
        
        sol = backtrack(limit, attempt_deadline)
        
        if sol:
            best_rows = sol
            best_M = max(max(r) for r in best_rows)
            limit = best_M - 1
        else:
            # Check if we timed out
            if time.time() > attempt_deadline - 0.5:
                break
            else:
                limit -= 1

    return best_rows
