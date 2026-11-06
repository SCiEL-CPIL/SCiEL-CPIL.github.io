"""Construct Sidon set using Integer Linear Programming."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Construct a Sidon set containing 0 with all differences 1..30, minimizing max element.
    Uses Integer Linear Programming (ILP) via scipy.optimize.milp.
    """
    start_time = time.time()
    max_time = 115
    
    # We iterate M from a lower bound upwards.
    # A Sidon set of size k has k(k-1)/2 differences.
    # We need to cover 30 differences.
    # k=9 -> 36 diffs.
    # k=8 -> 28 diffs (not enough).
    # So min size is 9.
    # We try to find the smallest M.
    
    for M in range(30, 200):
        if time.time() - start_time > max_time:
            break
            
        # Variables:
        # x_i for i in 0..M (M+1 vars)
        # z_{i,j} for 0 <= i < j <= M (M(M+1)/2 vars)
        
        n_x = M + 1
        n_pairs = M * (M + 1) // 2
        n_vars = n_x + n_pairs
        
        # Map pair (i, j) to index
        # Index of z_{i,j} = n_x + i * (M + 1) + j - (i * (i + 1)) // 2 
        # Simpler: just create a list of pairs and map index.
        pairs = []
        pair_indices = {}
        idx = n_x
        for i in range(M + 1):
            for j in range(i + 1, M + 1):
                pair_indices[(i, j)] = idx
                pairs.append((i, j))
                idx += 1
        
        # Objective: Minimize 0 (feasibility problem)
        c = np.zeros(n_vars)
        
        # Bounds
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        # x_0 must be 1
        lb[0] = 1.0
        ub[0] = 1.0
        
        # Constraints
        A_rows = []
        lb_rows = []
        ub_rows = []
        
        # 1. Sum x_i <= 32
        row = np.zeros(n_vars)
        for i in range(n_x):
            row[i] = 1.0
        A_rows.append(row)
        lb_rows.append(0.0)
        ub_rows.append(32.0)
        
        # 2. Linearization constraints for z_{i,j}
        # z_{i,j} <= x_i  =>  z_{i,j} - x_i <= 0
        # z_{i,j} <= x_j  =>  z_{i,j} - x_j <= 0
        # z_{i,j} >= x_i + x_j - 1  =>  -z_{i,j} + x_i + x_j >= 1
        
        for (i, j), z_idx in pair_indices.items():
            # z - x_i <= 0
            row = np.zeros(n_vars)
            row[z_idx] = 1.0
            row[i] = -1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
            
            # z - x_j <= 0
            row = np.zeros(n_vars)
            row[z_idx] = 1.0
            row[j] = -1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
            
            # -z + x_i + x_j >= 1  =>  z - x_i - x_j <= -1
            row = np.zeros(n_vars)
            row[z_idx] = 1.0
            row[i] = -1.0
            row[j] = -1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(-1.0)
            
        # 3. Sidon property: Each difference d occurs at most once.
        # For each d in 1..M:
        # sum_{i=0}^{M-d} z_{i, i+d} <= 1
        for d in range(1, M + 1):
            row = np.zeros(n_vars)
            count = 0
            for i in range(0, M - d + 1):
                j = i + d
                z_idx = pair_indices[(i, j)]
                row[z_idx] = 1.0
                count += 1
            if count > 0:
                A_rows.append(row)
                lb_rows.append(0.0)
                ub_rows.append(1.0)
                
        # 4. Coverage: Differences 1..30 must occur at least once.
        # For each d in 1..30:
        # sum_{i=0}^{M-d} z_{i, i+d} >= 1
        for d in range(1, 31):
            if d > M:
                # Impossible to cover d if M < d
                continue
            row = np.zeros(n_vars)
            count = 0
            for i in range(0, M - d + 1):
                j = i + d
                z_idx = pair_indices[(i, j)]
                row[z_idx] = 1.0
                count += 1
            if count > 0:
                A_rows.append(row)
                lb_rows.append(1.0)
                ub_rows.append(np.inf)
            else:
                # No pairs with difference d, infeasible
                break
        
        # Construct A matrix
        A = np.array(A_rows)
        lb_arr = np.array(lb_rows)
        ub_arr = np.array(ub_rows)
        
        # Solve
        try:
            res = milp(c=c, constraints=LinearConstraint(A, lb_arr, ub_arr), integrality=np.ones(n_vars), bounds=Bounds(lb, ub))
        except Exception:
            continue
            
        if res.status == 0:
            # Extract solution
            x_vals = res.x[:n_x]
            sol = [i for i in range(n_x) if x_vals[i] > 0.5]
            if 0 in sol:
                # Verify
                diffs = set()
                n = len(sol)
                sol = sorted(sol)
                for i in range(n):
                    for j in range(i + 1, n):
                        diffs.add(sol[j] - sol[i])
                if all(d in diffs for d in range(1, 31)) and len(diffs) == n*(n-1)//2:
                    return sol

    # Fallback if ILP fails or times out
    # Use a known valid set. 
    # From literature or quick search, a set covering 1..30 with small max.
    # Let's try to construct one greedily or use a hardcoded one if we know it.
    # A known Sidon set for 1..30 is {0, 1, 4, 10, 12, 17, 22, 28, 34} was invalid.
    # Let's use a larger valid set as fallback.
    a = [0]
    def diffs_list(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in set(diffs_list(sorted(a))):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                    a = trial
                    break
                y += 1
    return a
