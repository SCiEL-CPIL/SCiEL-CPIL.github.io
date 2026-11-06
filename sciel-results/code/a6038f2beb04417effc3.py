import numpy as np
from math import gcd
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 512
    candidates = list(range(1, N))
    n = len(candidates)
    
    # Weights: 1 / (512 - a)
    weights = np.array([1.0 / (N - a) for a in candidates])
    
    # Find conflicts (non-coprime pairs)
    # We need x_i + x_j <= 1 for all i < j such that gcd(candidates[i], candidates[j]) > 1
    rows = []
    cols = []
    
    # Optimization: Precompute gcds or just iterate. 511 is small.
    for i in range(n):
        ai = candidates[i]
        for j in range(i + 1, n):
            aj = candidates[j]
            if gcd(ai, aj) > 1:
                rows.append(i)
                rows.append(j)
                cols.append(1)
                cols.append(1)
                # We need to structure this for LinearConstraint
                # Actually, LinearConstraint takes A, lb, ub.
                # It's easier to build the matrix directly if dense, but it's sparse.
                # scipy.sparse is better.
    
    # Let's use sparse matrix
    from scipy.sparse import lil_matrix
    
    # Number of constraints
    num_constraints = len(rows) // 2
    if num_constraints == 0:
        # All are coprime? Unlikely.
        return candidates
        
    A = lil_matrix((num_constraints, n))
    
    # Re-iterate to fill A efficiently or use coo
    # The previous loop was inefficient for building sparse matrix directly.
    # Let's collect (row, col, val) tuples.
    
    conflict_tuples = []
    for i in range(n):
        ai = candidates[i]
        for j in range(i + 1, n):
            aj = candidates[j]
            if gcd(ai, aj) > 1:
                conflict_tuples.append((len(conflict_tuples), i, 1))
                conflict_tuples.append((len(conflict_tuples)-1, j, 1)) # Row index is same for both cols in a constraint
                
    # Wait, the row index for the constraint is the index of the constraint, not the tuple index.
    # Let's do this properly.
    
    constraint_rows = []
    constraint_cols = []
    constraint_vals = []
    
    constraint_idx = 0
    for i in range(n):
        ai = candidates[i]
        for j in range(i + 1, n):
            aj = candidates[j]
            if gcd(ai, aj) > 1:
                constraint_rows.append(constraint_idx)
                constraint_cols.append(i)
                constraint_vals.append(1)
                
                constraint_rows.append(constraint_idx)
                constraint_cols.append(j)
                constraint_vals.append(1)
                
                constraint_idx += 1
                
    if constraint_idx == 0:
        return candidates
        
    A = lil_matrix((constraint_idx, n))
    for r, c, v in zip(constraint_rows, constraint_cols, constraint_vals):
        A[r, c] = v
        
    A = A.tocsr()
    
    # Objective: Minimize -sum(w * x)
    c = -weights
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Constraints: 1 <= x_i + x_j <= 1  (since x are 0/1, this is x_i + x_j <= 1)
    # Actually, LinearConstraint(A, lb, ub)
    # We want A @ x <= 1
    # So lb = -inf, ub = 1? Or lb=0, ub=1?
    # Since x >= 0, x_i + x_j >= 0 is trivial.
    # So lb = 0, ub = 1.
    
    constraints = LinearConstraint(A, lb=0, ub=1)
    
    # Integrality: all integer
    integrality = np.ones(n)
    
    # Solve
    try:
        # options for HiGHS
        # time_limit in seconds
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 100})
        
        if res.status == 0:  # Optimal
            x = res.x
            # x might be float, round to int
            selected = [candidates[i] for i in range(n) if x[i] > 0.5]
            # Verify pairwise coprime
            valid = True
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if gcd(selected[i], selected[j]) != 1:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                return selected
            else:
                # Fallback if solver returned invalid due to precision?
                pass
                
        elif res.status == 1:  # Infeasible? Should not happen (empty set is feasible)
            pass
            
        elif res.status == 2:  # Unbounded? No.
            pass
            
        elif res.status == 3:  # Iteration limit? Or Time limit?
            # Check if we have a feasible solution (incumbent)
            if res.x is not None and len(res.x) == n:
                 x = res.x
                 selected = [candidates[i] for i in range(n) if x[i] > 0.5]
                 # Verify
                 valid = True
                 for i in range(len(selected)):
                     for j in range(i+1, len(selected)):
                         if gcd(selected[i], selected[j]) != 1:
                             valid = False
                             break
                     if not valid: break
                 
                 if valid:
                     return selected
                     
    except Exception as e:
        pass
        
    # Fallback: Greedy Heuristic
    # Sort by weight descending
    indices = np.argsort(-weights)
    chosen = []
    for i in indices:
        a = candidates[i]
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            
    return chosen
