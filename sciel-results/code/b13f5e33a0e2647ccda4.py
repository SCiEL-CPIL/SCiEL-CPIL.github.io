import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    K = k + 1
    
    # 1. Generate baseline using Greedy
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < K:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_A = greedy_construct()
    best_scope = max(r[-1] for r in best_A)
    
    # Helper to compute cost
    PENALTY = 100000
    def compute_diff_count(A):
        diff_count = {}
        for r in range(n_rows):
            row = A[r]
            for i in range(K):
                for j in range(i + 1, K):
                    d = row[j] - row[i]
                    diff_count[d] = diff_count.get(d, 0) + 1
        return diff_count

    def get_cost(A, diff_count):
        excess = sum(c - 1 for c in diff_count.values() if c > 1)
        scope = max(A[r][K-1] for r in range(n_rows))
        if excess > 0:
            return excess * PENALTY + scope
        else:
            return scope

    best_diff_count = compute_diff_count(best_A)
    best_cost = get_cost(best_A, best_diff_count)

    # 2. SA Loop
    while time.time() < deadline:
        # Initialize current state
        # Strategy: Perturb best_A
        A = [row[:] for row in best_A]
        
        # Apply random perturbations
        # Change some elements
        for _ in range(5):
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, K - 1)
            delta = random.randint(-10, 10)
            new_val = A[r][j] + delta
            # Ensure basic validity (increase)
            if j > 0 and new_val <= A[r][j-1]: continue
            if j < K - 1 and new_val >= A[r][j+1]: continue
            A[r][j] = new_val
            
        diff_count = compute_diff_count(A)
        cost = get_cost(A, diff_count)
        
        if cost < best_cost:
            best_cost = cost
            best_A = [row[:] for row in A]
            best_diff_count = diff_count # Copy? No, we recompute later.
            # Actually, we should copy diff_count if we want to be safe, 
            # but we recompute at the start of the next iteration anyway.
            
        T = 5.0
        cooling = 0.995
        steps = 0
        max_steps = 2000
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, K - 1)
            
            if random.random() < 0.1:
                delta = random.randint(-20, 20)
            else:
                delta = random.randint(-5, 5)
                
            old_val = A[r][j]
            new_val = old_val + delta
            
            if j > 0 and new_val <= A[r][j-1]:
                continue
            if j < K - 1 and new_val >= A[r][j+1]:
                continue
            
            # Update diff_count
            old_diffs = []
            for i in range(j):
                d = A[r][j] - A[r][i]
                old_diffs.append(d)
            for i in range(j + 1, K):
                d = A[r][i] - A[r][j]
                old_diffs.append(d)
                
            new_diffs = []
            for i in range(j):
                d = new_val - A[r][i]
                new_diffs.append(d)
            for i in range(j + 1, K):
                d = A[r][i] - new_val
                new_diffs.append(d)
                
            for d in old_diffs:
                diff_count[d] -= 1
                if diff_count[d] == 0:
                    del diff_count[d]
            for d in new_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
                
            A[r][j] = new_val
            new_cost = get_cost(A, diff_count)
            
            if new_cost < cost:
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_A = [row[:] for row in A]
                    # Note: best_diff_count is not updated here, but we don't use it for best_A logic 
                    # except for initialization of next restart. 
                    # We should update best_diff_count if we want to perturb from it accurately?
                    # Yes.
                    # But copying a dict is cheap.
                    pass 
            else:
                prob = math.exp(-(new_cost - cost) / T)
                if random.random() < prob:
                    cost = new_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_A = [row[:] for row in A]
                else:
                    # Revert
                    A[r][j] = old_val
                    for d in new_diffs:
                        diff_count[d] -= 1
                        if diff_count[d] == 0:
                            del diff_count[d]
                    for d in old_diffs:
                        diff_count[d] = diff_count.get(d, 0) + 1
            T *= cooling
            
    return best_A
