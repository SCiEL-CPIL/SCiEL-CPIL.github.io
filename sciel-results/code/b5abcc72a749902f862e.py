import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, leaving 10s margin for overhead/evaluation
    
    n = 256
    # 1. Identify constraints
    # A conflict exists if a + b is a perfect square.
    # We need x_a + x_b <= 1 for all conflicting pairs (a, b).
    
    # Precompute perfect squares up to 2*256 = 512
    max_sum = 2 * n
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # Build the list of constraints for the ILP
    # We will use scipy.optimize.milp
    # Variables: x[0] ... x[255] corresponding to 1 ... 256
    
    constraints = []
    
    # Diagonal constraints: if 2*i is a square, then 2*x_i <= 1 => x_i <= 0.5 => x_i = 0
    # Off-diagonal: if i+j is a square, x_i + x_j <= 1
    
    # It is more efficient to build the sparse matrix A_ub and b_ub directly
    # But milp takes LinearConstraint objects.
    # Let's collect rows for the constraint matrix.
    
    rows = []
    cols = []
    vals = []
    lbs = []
    ubs = []
    
    # We only need upper bounds of 1 for these sum constraints.
    # Lower bounds are 0 (implicitly, since x >= 0).
    
    # Check diagonals
    for i in range(1, n + 1):
        if 2 * i in squares:
            # 2 * x_i <= 1
            rows.append([2 * i])
            cols.append([i - 1]) # 0-indexed
            vals.append([2])
            lbs.append([0])
            ubs.append([1])
    
    # Check off-diagonals
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if i + j in squares:
                # x_i + x_j <= 1
                rows.append([i, j])
                cols.append([i - 1, j - 1])
                vals.append([1, 1])
                lbs.append([0])
                ubs.append([1])
    
    if not rows:
        # No constraints? Just return all
        return list(range(1, n + 1))
    
    # Convert to sparse format for efficiency if possible, or dense if small enough
    # n=256, max constraints ~ 256*255/2 = 32k. Dense is 32k x 256 floats = 8M floats ~ 64MB. Manageable.
    # However, sparse is safer.
    
    # scipy.optimize.milp accepts LinearConstraint with sparse or dense A.
    # Let's build a sparse matrix.
    import scipy.sparse as sp
    
    # Flatten lists for sparse matrix construction
    row_idx = []
    col_idx = []
    data = []
    
    for r_idx, (r, c, v) in enumerate(zip(rows, cols, vals)):
        for cc, vv in zip(c, v):
            row_idx.append(r_idx)
            col_idx.append(cc)
            data.append(vv)
            
    A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(len(rows), n))
    
    # Objective: maximize sum(x) => minimize -sum(x)
    c = -np.ones(n)
    
    # Constraints: 0 <= A x <= 1
    lb = np.zeros(len(rows))
    ub = np.ones(len(rows))
    
    constraint = LinearConstraint(A, lb, ub)
    
    # Bounds: 0 <= x <= 1
    integrality = np.ones(n) # All integer
    bounds = Bounds(0, 1)
    
    # Solve
    try:
        res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            # x is float array, check > 0.5
            selected = [int(i + 1) for i in range(n) if x[i] > 0.5]
            if selected:
                return selected
            else:
                # Fallback if solver returns empty (shouldn't happen for max independent set on non-empty graph)
                return [1]
        else:
            # If solver fails, fall back to heuristic
            pass
    except Exception:
        pass
    
    # Fallback Heuristic: Greedy
    # Simple Greedy Fallback
    used = set()
    result = []
    for i in range(1, n + 1):
        if i in used:
            continue
        # Check if i conflicts with any in result
        ok = True
        for j in result:
            if i + j in squares:
                ok = False
                break
        if ok:
            # Check diagonal
            if 2 * i in squares:
                continue
            result.append(i)
            # Mark conflicts? Not necessary for correctness, just speed.
            # For n=256, O(N^2) is fine.
            
    if result:
        return result
    return [1]

# EVOLVE-BLOCK-END
