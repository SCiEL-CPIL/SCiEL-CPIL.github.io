import time

def construct(n_rows, k):
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    M_min = 1
    while M_min * (M_min + 1) // 2 < total_diffs_needed:
        M_min += 1
        
    start_time = time.time()
    time_limit = 115
    
    # We will try M from M_min upwards
    # To avoid getting stuck on infeasible M, we limit the number of nodes explored per M
    # or the time spent per M.
    
    def solve_with_limit(M, time_budget):
        rows = []
        used_diffs = set()
        nodes = [0]
        start_t = time.time()
        
        def backtrack(row_idx, current_row):
            # Time check
            if time.time() - start_t > time_budget:
                return False
            
            nodes[0] += 1
            # Optional: node limit
            # if nodes[0] > 100000: return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Check symmetry: current_row must be >= previous row
                if row_idx > 0:
                    if current_row[1:] < rows[row_idx-1][1:]:
                        return False
                rows.append(current_row)
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_mark = current_row[-1]
            # Heuristic: Start from last_mark + 1
            # We can also try to jump ahead if we know we need to fill k more marks
            # and the max allowed is M.
            # Min possible next mark is last_mark + 1.
            # Max possible next mark is M - (k - len(current_row) + 1) ? No, just M.
            
            for x in range(last_mark + 1, M + 1):
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, current_row):
                    return True
                
                current_row.pop()
                used_diffs.difference_update(new_diffs)
                
            return False
        
        if backtrack(0, [0]):
            return rows
        return None

    # Strategy:
    # 1. Try M from M_min to M_min + 20 with a small time budget each (e.g., 2s).
    # 2. If found, return.
    # 3. If not, try larger M with remaining time?
    # Or, just try M=112 (known best) first?
    # The prompt says "best known ... 112".
    # If we find a solution with 112, it's a good candidate.
    # But we want the MINIMUM.
    # So we must check smaller M.
    
    # Let's iterate M.
    # To be safe, we'll use a global time check.
    
    for M in range(M_min, 150):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Allocate time for this M
        # If M is close to M_min, give it more time?
        # Or just a fixed small time, e.g., 1s?
        # 115s / 45 M values = ~2.5s per M.
        time_budget = 2.0
        
        sol = solve_with_limit(M, time_budget)
        if sol is not None:
            return sol
            
    # Fallback
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
