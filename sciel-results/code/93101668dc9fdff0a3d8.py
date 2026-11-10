def construct(n_rows, k):
    import time
    start_time = time.time()
    TIME_LIMIT = 115
    
    total_diffs_per_row = k * (k + 1) // 2
    LB = n_rows * total_diffs_per_row
    
    def try_scope(M):
        used = [False] * (M + 1)
        rows = []
        
        def lex_ge(row1, row2):
            for a, b in zip(row1, row2):
                if a < b:
                    return False
                if a > b:
                    return True
            return len(row1) >= len(row2)
            
        def find_row(min_row, max_scope):
            def dfs(current_row, current_diffs):
                if time.time() - start_time > TIME_LIMIT:
                    return None
                if len(current_row) == k + 1:
                    if min_row is None or lex_ge(current_row, min_row):
                        return current_row[:]
                    return None
                
                last = current_row[-1]
                remaining = k + 1 - len(current_row)
                
                if max_scope - last < remaining:
                    return None
                
                min_next = last + 1
                
                # Pruning based on min_row (lexicographic constraint)
                if min_row is not None:
                    is_prefix = True
                    for i in range(len(current_row)):
                        if current_row[i] != min_row[i]:
                            is_prefix = False
                            if current_row[i] < min_row[i]:
                                return None
                            else:
                                break
                    if is_prefix and len(current_row) < len(min_row):
                        min_next = max(min_next, min_row[len(current_row)])
                
                for next_val in range(min_next, max_scope + 1):
                    valid = True
                    new_diffs = []
                    for a in current_row:
                        d = next_val - a
                        if d > M or used[d] or d in current_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    current_diffs.update(new_diffs)
                    current_row.append(next_val)
                    
                    result = dfs(current_row, current_diffs)
                    if result is not None:
                        return result
                    
                    current_row.pop()
                    current_diffs.difference_update(new_diffs)
                
                return None
            
            return dfs([0], set())
        
        def backtrack(row_idx, prev_row):
            if time.time() - start_time > TIME_LIMIT:
                return False
            if row_idx == n_rows:
                return True
            
            row = find_row(prev_row, M)
            if row is None:
                return False
            
            new_diffs = []
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    new_diffs.append(d)
            
            for d in new_diffs:
                if used[d]:
                    return False
                used[d] = True
            
            rows.append(row)
            
            if backtrack(row_idx + 1, row):
                return True
            
            for d in new_diffs:
                used[d] = False
            rows.pop()
            
            return False
        
        if backtrack(0, None):
            return rows
        return None

    for M in range(LB, LB + 100):
        result = try_scope(M)
        if result is not None:
            return result
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
    # Fallback
    used_set = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used_set:
                row.append(x)
                used_set |= new
        rows.append(row)
    return rows
