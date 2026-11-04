import random
import time
import sys

# Increase recursion limit just in case
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Greedy baseline to get an initial upper bound on scope
    def greedy_solution():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if not conflict:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_solution()
    best_scope = max(max(r) for r in best_rows)
    
    # Lower bound on scope
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    
    # Randomized Backtracking Solver
    # We try to find a solution with scope <= M
    def solve_with_scope(M, seed=None):
        if seed is not None:
            random.seed(seed)
        
        used = 0
        rows = []
        
        def build_row(row_idx, current_row, used_mask):
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if build_row(row_idx + 1, [0], used_mask):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            # Candidates for next value: last_val + 1 to M
            candidates = list(range(last_val + 1, M + 1))
            
            # Filter valid candidates
            valid_candidates = []
            for x in candidates:
                new_diffs = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                
                if not conflict:
                    valid_candidates.append((x, new_diffs))
            
            if not valid_candidates:
                return False
            
            # Randomize order to escape local minima
            random.shuffle(valid_candidates)
            
            for x, new_diffs in valid_candidates:
                if build_row(row_idx, current_row + [x], used_mask | new_diffs):
                    return True
            
            return False
        
        if build_row(0, [0], 0):
            return rows
        return None

    # Try to improve the scope
    # We start from the greedy scope - 1 and go down
    attempts_per_scope = 10
    
    current_scope = best_scope
    min_scope = lower_bound
    
    for M in range(current_scope - 1, min_scope - 1, -1):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(attempts_per_scope):
            if time.time() > deadline:
                break
                
            seed = M * 1000 + attempt
            res = solve_with_scope(M, seed)
            if res:
                # Update best
                best_rows = res
                current_scope = M
                found = True
                break
        
        if not found:
            # If we fail to find a solution for M, 
            # it's likely infeasible or very hard. 
            # Since M-1 is harder, we break.
            break
            
    # Final validation
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    if not validate(best_rows):
        # Fallback to pure greedy if something went wrong
        best_rows = greedy_solution()
        best_scope = max(max(r) for r in best_rows)

    return best_rows
