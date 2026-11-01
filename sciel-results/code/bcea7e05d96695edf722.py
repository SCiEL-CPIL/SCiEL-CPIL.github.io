import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0
    
    # 1. Deterministic Greedy
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                new_diffs = 0
                for a in row:
                    d = x - a
                    if d > 0:
                        bit = 1 << d
                        if used & bit:
                            valid = False
                            break
                        new_diffs |= bit
                if valid:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, max(max(r) for r in rows)
    
    best_rows, best_M = greedy()
    
    # 2. Randomized Greedy
    for _ in range(200):
        if time.time() - start_time > time_limit - 10:
            break
        used = 0
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                for dx in range(1, 15):
                    candidate = x + dx
                    if candidate >= best_M:
                        break
                    valid = True
                    new_diffs = 0
                    for a in row:
                        d = candidate - a
                        if d > 0:
                            bit = 1 << d
                            if used & bit:
                                valid = False
                                break
                            new_diffs |= bit
                    if valid:
                        x = candidate
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    success = False
                    break
            if not success:
                break
            rows.append(row)
        if success:
            m = max(max(r) for r in rows)
            if m < best_M:
                best_M = m
                best_rows = rows

    # 3. DFS for improvement
    lower_bound = n_rows * (k * (k - 1) // 2)
    
    def find_solution(M):
        used = 0
        rows = []
        
        def dfs(row_idx, current_row, prev_max):
            if row_idx == n_rows:
                return True
            
            if not current_row:
                current_row = [0]
                
            if len(current_row) == k + 1:
                if current_row[-1] > M:
                    return False
                rows.append(current_row)
                if dfs(row_idx + 1, [], current_row[-1]):
                    return True
                rows.pop()
                return False
            
            last_mark = current_row[-1]
            remaining_marks = (k + 1) - len(current_row) - 1
            
            for candidate in range(last_mark + 1, M + 1):
                if candidate + remaining_marks > M:
                    break
                
                valid = True
                new_diffs = 0
                for a in current_row:
                    d = candidate - a
                    if d > 0:
                        bit = 1 << d
                        if used & bit:
                            valid = False
                            break
                        new_diffs |= bit
                
                if valid:
                    used |= new_diffs
                    current_row.append(candidate)
                    
                    if dfs(row_idx, current_row, prev_max):
                        return True
                    
                    current_row.pop()
                    used ^= new_diffs
                    
            return False
        
        if dfs(0, [], 0):
            return rows
        return None

    # Try to find a solution with smaller M
    for M in range(best_M - 1, lower_bound - 1, -1):
        if time.time() - start_time > time_limit - 5:
            break
        sol = find_solution(M)
        if sol:
            return sol

    return best_rows
