import time
from math import lcm, gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = 50.0 # Safety margin before 60s limit (evaluator kills at 90s, but keep it tight)
    
    # Range is [2, 256]
    nums = list(range(2, 257))
    n = len(nums)
    
    # Objective: Maximize sum(1/x)
    # milp minimizes c @ x, so we minimize -sum(1/x)
    c = -np.array([1.0 / x for x in nums])
    
    # Constraints: x_i + x_j <= 1 if lcm(i, j) <= 256
    # We need to build the sparse matrix A for these constraints.
    rows = []
    cols = []
    data = []
    
    # Precompute constraints
    # There are O(n^2) pairs, n=255, so ~32k pairs.
    # This loop is fast.
    for i in range(n):
        for j in range(i + 1, n):
            a = nums[i]
            b = nums[j]
            # Check if they conflict
            # lcm(a, b) = a * b // gcd(a, b)
            # We want to enforce x_i + x_j <= 1 if lcm(a, b) <= 256
            # Note: if lcm > 256, no constraint needed.
            g = gcd(a, b)
            l = a * b // g
            if l <= 256:
                rows.append(len(data))
                cols.append(i)
                data.append(1.0)
                
                rows.append(len(data))
                cols.append(j)
                data.append(1.0)
    
    # Create sparse matrix
    A = csr_matrix((data, (rows, cols)), shape=(len(data) // 2, n))
    
    # Constraints: 0 <= x_i <= 1 and A x <= 1
    # LinearConstraint(A_ub, b_ub, A_eq, b_eq)
    # Here we only have inequality constraints A x <= 1
    # And bounds 0 <= x <= 1
    constraints = LinearConstraint(A, ub=np.ones(A.shape[0]))
    integrality = np.ones(n)
    bounds = Bounds(lb=0, ub=1)
    
    # Solve
    res = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        method="highs"
    )
    
    # Check time
    if time.time() > deadline:
         # Fallback to seed-like greedy if solver is slow (unlikely)
         selected = []
         for x in nums:
             if all(lcm(x, y) > 256 for y in selected):
                 selected.append(x)
         return selected

    if res.success:
        # res.x contains the optimal values (0 or 1)
        # Small epsilon check for robustness
        mask = res.x > 0.5
        selected = [nums[i] for i in range(n) if mask[i]]
        
        # Verify validity just in case
        valid = True
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if lcm(selected[i], selected[j]) <= 256:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return selected
        else:
            # If validation fails (shouldn't happen), fallback
            selected = []
            for x in nums:
                if all(lcm(x, y) > 256 for y in selected):
                    selected.append(x)
            return selected
    else:
        # Fallback to greedy
        selected = []
        for x in nums:
            if all(lcm(x, y) > 256 for y in selected):
                selected.append(x)
        return selected

