import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
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

    def backtrack(limit):
        used = 0
        rows = []
        
        def dfs(row_idx, current_row, last_val, used_mask, prev_row):
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return rows[:]
            
            if len(current_row) == k + 1:
                if prev_row is not None:
                    if current_row <= prev_row:
                        return None
                
                rows.append(current_row)
                res = dfs(row_idx + 1, [0], 0, used_mask, current_row)
                if res:
                    return res
                rows.pop()
                return None
            
            start = last_val + 1
            
            for x in range(start, limit + 1):
                new_mask = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if d > limit:
                        conflict = True
                        break
                    bit = 1 << d
                    if used_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                remaining_slots = k - len(current_row)
                if x + remaining_slots > limit:
                    break
                
                res = dfs(row_idx, current_row + [x], x, used_mask | new_mask, prev_row)
                if res:
                    return res
            
            return None
        
        return dfs(0, [0], 0, 0, None)

    rows_greedy = greedy()
    best_rows = rows_greedy
    best_M = max(max(r) for r in best_rows)
    
    limit = best_M - 1
    
    while limit > 0 and time.time() < deadline:
        sol = backtrack(limit)
        if sol:
            best_rows = sol
            best_M = max(max(r) for r in best_rows)
            limit = best_M - 1
        else:
            break

    return best_rows
