import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for cleanup/return
    
    # Calculate lower bound for scope M
    # Total differences needed: N = n_rows * k * (k+1) / 2
    N = n_rows * k * (k + 1) // 2
    M_min = int(math.ceil((-1 + math.sqrt(1 + 8 * N)) / 2))
    
    # Get a valid solution using greedy to establish an upper bound M_max
    # and as a fallback
    greedy_rows = greedy_construct(n_rows, k)
    M_max = max(max(row) for row in greedy_rows)
    
    # If greedy is already at the lower bound, return it
    if M_max == M_min:
        return greedy_rows
        
    # Try to find a solution with scope M in [M_min, M_max]
    # We prioritize smaller M.
    for M in range(M_min, M_max + 1):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Allocate remaining time to this search attempt
        # We don't want to spend all time on one M if it's hard.
        # But if we find a solution, we return immediately.
        # If we don't, we move to next M.
        # Let's give it the remaining time, but cap it if we have many M's left?
        # No, if M_min is infeasible, we want to try M_min+1.
        # If we spend 100s on M_min and fail, we have 15s left.
        # We can try M_min+1 with 15s.
        
        result = randomized_backtrack(n_rows, k, M, time_limit - elapsed)
        if result is not None:
            return result
            
    # If we didn't find a better one, or ran out of time, return greedy
    return greedy_rows

def greedy_construct(n_rows, k):
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
                used.update(new_diffs)
        rows.append(row)
    return rows

def randomized_backtrack(n_rows, k, M, time_budget):
    start_time = time.time()
    used_diffs = set()
    rows = []
    
    def backtrack(row_index, current_row):
        if time.time() - start_time > time_budget:
            return False
            
        if row_index == n_rows:
            return True
            
        if len(current_row) == k + 1:
            rows.append(current_row)
            return backtrack(row_index + 1, [0])
            
        last_val = current_row[-1]
        
        candidates = []
        # Optimization: Only check x that are > last_val and <= M
        for x in range(last_val + 1, M + 1):
            valid = True
            for a in current_row:
                d = x - a
                if d > M or d in used_diffs:
                    valid = False
                    break
            if valid:
                candidates.append(x)
                
        if not candidates:
            return False
            
        random.shuffle(candidates)
        
        for x in candidates:
            new_diffs = {x - a for a in current_row}
            
            current_row.append(x)
            used_diffs.update(new_diffs)
            
            if backtrack(row_index, current_row):
                return True
                
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False
        
    return backtrack(0, [0])
