import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110
    
    # Calculate lower bound based on counting argument
    # Total distinct differences required = n_rows * (k * (k + 1) // 2)
    total_diffs = n_rows * (k * (k + 1) // 2)
    lower_bound = total_diffs
    
    def backtrack(M):
        used = set()
        rows = []
        timeout_flag = [False]
        
        def dfs(row_idx, col_idx, current_row):
            if time.time() - start_time > time_limit:
                timeout_flag[0] = True
                return None
                
            if timeout_flag[0]:
                return None

            if row_idx == n_rows:
                if col_idx == 0:
                    return rows[:]
                else:
                    return None
            
            if col_idx == k: # Finished current row
                # Symmetry breaking: enforce lexicographic order of rows
                if row_idx > 0:
                    if current_row < rows[-1]:
                        return None
                rows.append(current_row)
                res = dfs(row_idx + 1, 0, [0])
                if res is not None:
                    return res
                rows.pop()
                return None
            
            last_val = current_row[-1]
            start = last_val + 1
            end = M
            
            # Iterate through possible values
            # Heuristic: Try smaller values first
            for val in range(start, end + 1):
                diffs = {val - a for a in current_row}
                if diffs & used:
                    continue
                
                # Pruning: Check if remaining differences can fit
                new_used_count = len(used) + len(diffs)
                rem_diffs = total_diffs - new_used_count
                avail_slots = M - new_used_count
                if avail_slots < rem_diffs:
                    continue
                
                used.update(diffs)
                res = dfs(row_idx, col_idx + 1, current_row + [val])
                used.difference_update(diffs)
                
                if res is not None:
                    return res
                if timeout_flag[0]:
                    return None
                    
            return None
            
        return dfs(0, 0, [0])

    # Try M from lower_bound upwards
    # The best known is 112, lower bound is 105.
    # We try a small range to find the optimal or near-optimal solution.
    for M in range(lower_bound, lower_bound + 20):
        if time.time() - start_time > time_limit:
            break
        sol = backtrack(M)
        if sol is not None:
            return sol
            
    # Fallback: Greedy construction if backtracking fails or times out
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
