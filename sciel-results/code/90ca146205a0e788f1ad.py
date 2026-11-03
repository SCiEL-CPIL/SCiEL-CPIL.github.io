import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    def binom(n, r):
        if r > n: return 0
        if r == 0 or r == n: return 1
        return math.comb(n, r)
        
    total_diffs_per_row = binom(k+1, 2)
    
    # 1. Greedy
    def greedy_solution():
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

    best_sol = greedy_solution()
    best_max = max(max(row) for row in best_sol)
    
    # 2. Backtracking
    def solve_with_cap(cap, time_limit, shuffle=False):
        used_diffs = set()
        rows = [[0] * (k+1) for _ in range(n_rows)]
        
        def get_remaining_needed(r, c):
            curr_row_rem = binom(k+1, 2) - binom(c, 2)
            future_rows = n_rows - r - 1
            future_diffs = future_rows * total_diffs_per_row
            return curr_row_rem + future_diffs

        def backtrack(r, c, start_time):
            if time.time() - start_time > time_limit:
                return False
            
            if r == n_rows:
                return True
            
            if c == 0:
                rows[r][0] = 0
                return backtrack(r, 1, start_time)
            
            prev = rows[r][c-1]
            
            # Symmetry breaking: rows[r][1] >= rows[r-1][1]
            min_val = prev + 1
            if r > 0 and c == 1:
                min_val = max(min_val, rows[r-1][1])
            
            needed = get_remaining_needed(r, c)
            available = cap - len(used_diffs)
            if available < needed:
                return False
            
            candidates = list(range(min_val, cap + 1))
            if shuffle:
                random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - rows[r][i] for i in range(c)}
                if new_diffs & used_diffs:
                    continue
                
                used_diffs.update(new_diffs)
                rows[r][c] = x
                
                next_r, next_c = r, c + 1
                if next_c == k + 1:
                    next_r, next_c = r + 1, 0
                
                if backtrack(next_r, next_c, start_time):
                    return True
                
                used_diffs.difference_update(new_diffs)
                rows[r][c] = 0
            
            return False

        if backtrack(0, 0, time.time()):
            return [row[:] for row in rows]
        return None

    # Main Loop
    target = best_max - 1
    while target >= 112 and time.time() < deadline:
        found = False
        # Try up to 10 times for this target
        for attempt in range(10):
            if time.time() >= deadline: break
            slice_time = min(1.0, deadline - time.time())
            if slice_time <= 0: break
            
            sol = solve_with_cap(target, slice_time, shuffle=(attempt > 0))
            if sol:
                best_sol = sol
                best_max = max(max(row) for row in sol)
                target = best_max - 1
                found = True
                break
        
        if not found:
            target -= 1
            
    return best_sol
