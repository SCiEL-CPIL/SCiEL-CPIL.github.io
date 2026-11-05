import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Calculate lower bound for M (scope)
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    M_lower = total_diffs_needed

    # 2. Iterate possible scopes M
    # We try a range of M values starting from the lower bound
    max_search_range = 80  # Search up to M_lower + 80
    
    for M in range(M_lower, M_lower + max_search_range + 1):
        if time.time() > deadline:
            break
            
        # 3. Try to find a valid configuration for this M
        # Use randomized restarts
        num_restarts = 100
        for attempt in range(num_restarts):
            if time.time() > deadline:
                break
                
            result = try_construct_random(n_rows, k, M, attempt)
            if result is not None:
                return result
                
    # 4. Fallback to greedy construction (always valid, but suboptimal)
    return greedy_construct(n_rows, k)

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

def try_construct_random(n_rows, k, M, seed):
    random.seed(seed)
    used_diffs = set()
    rows = []
    
    for r in range(n_rows):
        # Quick feasibility check:
        # Are there enough unused differences in [1, M]?
        # We need diffs_per_row more.
        # Counting unused diffs in [1, M]
        # Since M is small, we can just count.
        available = 0
        for d in range(1, M + 1):
            if d not in used_diffs:
                available += 1
                if available >= k * (k + 1) // 2:
                    break
        
        if available < k * (k + 1) // 2:
            return None
            
        row = find_row_random(used_diffs, k, M, seed + r * 1000)
        if row is None:
            return None
            
        rows.append(row)
        # Update used differences
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
                
    return rows

def find_row_random(used_diffs, k, M, seed):
    random.seed(seed)
    row = [0]
    
    def dfs(current_row):
        if len(current_row) == k + 1:
            return current_row[:]
        
        last_val = current_row[-1]
        
        # Generate candidates
        # We want to try values that are likely to lead to a solution.
        # Smaller values are generally better for packing, but we are bounded by M.
        # Randomizing helps avoid local minima.
        
        candidates = list(range(last_val + 1, M + 1))
        random.shuffle(candidates)
        
        # Optional: Limit candidates to speed up failure?
        # If we don't find a row, we return None quickly.
        
        for next_val in candidates:
            new_diffs = set()
            valid = True
            for prev_val in current_row:
                d = next_val - prev_val
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                res = dfs(current_row + [next_val])
                if res is not None:
                    return res
        
        return None

    return dfs(row)
