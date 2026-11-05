import time
import numpy as np
from math import lcm, gcd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin before 90s kill

    # 1. Define the set of candidates
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # 2. Construct the conflict graph
    # We need constraints x_i + x_j <= 1 for all pairs (i, j) with lcm(i, j) <= 256
    # We will build a sparse matrix A for these constraints.
    
    rows = []
    cols = []
    data = []
    
    # Precompute LCMs or check on the fly?
    # On the fly is O(N^2) = 255^2 ~ 65k operations. Very fast.
    
    conflict_count = 0
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            # Calculate LCM
            # lcm(a, b) = a * b // gcd(a, b)
            g = gcd(vi, vj)
            l = vi * vj // g
            if l <= 256:
                # Conflict: cannot select both i and j
                # We need to add two entries to the sparse matrix for this constraint row
                # Row index is conflict_count
                rows.append(conflict_count)
                cols.append(i)
                data.append(1)
                
                rows.append(conflict_count)
                cols.append(j)
                data.append(1)
                
                conflict_count += 1
                
    # If no conflicts, we can select all. But there will be conflicts.
    # Construct sparse matrix
    m = conflict_count
    if m > 0:
        A = sparse.csr_matrix((data, (rows, cols)), shape=(m, n))
    else:
        A = sparse.csr_matrix((0, n))
        
    # 3. Objective function
    # Maximize sum(1/x * x_i)
    # Minimize -sum(1/x * x_i)
    c = np.array([-1.0 / v for v in candidates])
    
    # 4. Constraints
    # A @ x <= 1
    # Bounds for x: 0 <= x <= 1
    # Integrality: x is binary (0 or 1)
    
    constraints = LinearConstraint(A, ub=np.ones(m))
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n)
    
    # 5. Solve
    # Use HiGHS solver
    # Set a time limit just in case, though it should be fast
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 30})
        
        if not res.success:
            # Fallback: Greedy approach if MIP fails
            # Sort by 1/x descending (i.e., x ascending)
            # Pick x if it doesn't conflict with already picked
            selected = []
            for v in sorted(candidates):
                if all(lcm(v, y) > 256 for y in selected):
                    selected.append(v)
            return selected

        # Extract solution
        x_sol = res.x
        selected = [candidates[i] for i in range(n) if x_sol[i] > 0.5]
        
        # Verify validity (paranoid check)
        valid = True
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if lcm(selected[i], selected[j]) <= 256:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback
            selected = []
            for v in sorted(candidates):
                if all(lcm(v, y) > 256 for y in selected):
                    selected.append(v)
            return selected
            
        return selected
        
    except Exception:
        # Fallback: Greedy approach
        selected = []
        for v in sorted(candidates):
            if all(lcm(v, y) > 256 for y in selected):
                selected.append(v)
        return selected
