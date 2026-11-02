import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Fallback greedy solution
    def greedy():
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

    try:
        total_diffs = n_rows * k * (k + 1) // 2
        lower_bound = total_diffs
        
        def find_solution(m):
            rows = []
            used_diffs = set()
            
            def bt():
                if len(rows) == n_rows:
                    return True
                
                if time.time() > deadline:
                    return False
                
                prev_row = rows[-1] if rows else None
                
                # Build a row
                row = [0]
                used_in_row = set()
                
                def dfs_row():
                    if len(row) == k + 1:
                        if prev_row is not None and row <= prev_row:
                            return False
                        # Success
                        # Update global state
                        for i in range(k+1):
                            for j in range(i+1, k+1):
                                used_diffs.add(row[j] - row[i])
                        rows.append(row[:])
                        
                        if bt():
                            return True
                        
                        # Backtrack
                        rows.pop()
                        for i in range(k+1):
                            for j in range(i+1, k+1):
                                used_diffs.discard(row[j] - row[i])
                        return False
                    
                    if time.time() > deadline:
                        return False
                        
                    remaining = k + 1 - len(row)
                    if m - row[-1] < remaining:
                        return False
                    
                    # Determine start
                    start = row[-1] + 1
                    if prev_row is not None:
                        # Check prefix logic
                        is_prefix = True
                        for i in range(len(row)):
                            if row[i] != prev_row[i]:
                                is_prefix = False
                                if row[i] < prev_row[i]:
                                    return False
                                break
                        if is_prefix:
                            start = max(start, prev_row[len(row)] + 1)
                    
                    for x in range(start, m + 1):
                        # Check diffs
                        ok = True
                        for a in row:
                            d = x - a
                            if d in used_in_row or d in used_diffs:
                                ok = False
                                break
                        if not ok:
                            continue
                        
                        # Add x
                        row.append(x)
                        # Update used_in_row
                        for a in row[:-1]:
                            used_in_row.add(x - a)
                        
                        if dfs_row():
                            return True
                        
                        # Backtrack row
                        row.pop()
                        for a in row:
                            used_in_row.discard(x - a)
                    
                    return False
                
                return dfs_row()
            
            if bt():
                return rows
            return None
        
        # Search for minimal m
        # We can cap the search range to avoid infinite loops if lower_bound is wrong or time runs out
        # But lower_bound is a valid lower bound.
        # We will search up to lower_bound + 1000 or until time runs out.
        for m in range(lower_bound, lower_bound + 1000):
            if time.time() > deadline:
                break
            sol = find_solution(m)
            if sol:
                return sol
                
        # If no solution found in search, return greedy
        return greedy()
        
    except Exception:
        return greedy()
