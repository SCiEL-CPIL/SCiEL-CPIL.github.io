import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    # Lower bound on the maximum entry M
    lower_bound = n_rows * (k * (k + 1)) // 2
    
    # Fallback: Seed greedy
    def greedy_seed():
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

    best_rows = greedy_seed()
    best_M = max(max(r) for r in best_rows)
    
    # Randomized Backtracking Solver
    def try_solve(M, max_steps=100000):
        used_diffs = set()
        rows = []
        steps = 0
        failed = False
        
        def backtrack(row_idx, current_row, last_val, temp_used):
            nonlocal steps, failed
            if time.time() > deadline or failed:
                return False
            steps += 1
            if steps > max_steps:
                failed = True
                return False
                
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row.copy())
                if backtrack(row_idx + 1, [0], 0, set()):
                    return True
                rows.pop()
                return False
            
            candidates = []
            # Optimization: 
            # If the number of remaining elements is `rem`, the smallest possible span is `rem`.
            # So x + rem <= M is a necessary condition for the rest of the row to fit?
            # Actually, the next element must be at least x+1.
            # So the last element will be at least x + rem.
            # So x <= M - rem.
            rem = k + 1 - len(current_row)
            max_x = M - rem
            
            for x in range(last_val + 1, max_x + 1):
                new_diffs = {x - a for a in current_row}
                if not (new_diffs & used_diffs or new_diffs & temp_used):
                    candidates.append((x, new_diffs))
            
            if not candidates:
                return False
            
            # Heuristic: 80% smallest, 20% random
            if random.random() < 0.8:
                candidates.sort(key=lambda t: t[0])
            else:
                random.shuffle(candidates)
            
            for x, new_diffs in candidates:
                if backtrack(row_idx, current_row + [x], x, temp_used | new_diffs):
                    return True
            
            return False
        
        if backtrack(0, [0], 0, set()):
            return rows
        return None

    # Search for better M
    max_restarts_per_M = 3
    
    for M in range(lower_bound, best_M):
        if time.time() > deadline:
            break
        
        found_rows = None
        for _ in range(max_restarts_per_M):
            if time.time() > deadline:
                break
            res = try_solve(M)
            if res is not None:
                found_rows = res
                break
        
        if found_rows is not None:
            best_rows = found_rows
            best_M = M
            if M == lower_bound:
                break

    return best_rows
# EVOLVE-BLOCK-END
