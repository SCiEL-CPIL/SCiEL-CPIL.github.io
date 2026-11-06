import math
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 512
    
    # Weights for a in 1..511
    # We will solve for 2..511 and add 1 manually
    # a=1 is coprime to all, weight 1/511. Always beneficial to include.
    
    candidates = list(range(2, n))
    m = len(candidates)
    
    # Weights
    weights = np.array([1.0 / (n - a) for a in candidates])
    
    # Objective: Minimize -sum(w_i * x_i)
    c = -weights
    
    # Build conflict constraints
    # x_i + x_j <= 1 for gcd(candidates[i], candidates[j]) > 1
    
    # To be efficient, we can precompute conflicts
    # But building a sparse matrix for 50k constraints x 510 vars is fine.
    
    # List of (i, j) conflicts
    conflict_pairs = []
    for i in range(m):
        a = candidates[i]
        for j in range(i + 1, m):
            b = candidates[j]
            if gcd(a, b) > 1:
                conflict_pairs.append((i, j))
                
    # If no conflicts, we just pick all? No, gcd(a,b)>1 means conflict.
    # If list is empty, no constraints.
    
    # Construct A matrix (sparse)
    # Each constraint is x_i + x_j <= 1
    # A row has 1 at i and 1 at j.
    
    if conflict_pairs:
        import scipy.sparse as sp
        rows = []
        cols = []
        data = []
        for i, j in conflict_pairs:
            rows.append(len(rows)) # Row index
            cols.append(i)
            data.append(1.0)
            rows.append(len(rows)) # Wait, this logic is wrong for COO construction directly like this.
            
        # Better to build COO components
        row_idx = []
        col_idx = []
        data_vals = []
        
        for k, (i, j) in enumerate(conflict_pairs):
            row_idx.append(k)
            col_idx.append(i)
            data_vals.append(1.0)
            
            row_idx.append(k)
            col_idx.append(j)
            data_vals.append(1.0)
            
        A = sp.coo_matrix((data_vals, (row_idx, col_idx)), shape=(len(conflict_pairs), m)).tocsr()
        b_ub = np.ones(len(conflict_pairs))
        constraint = LinearConstraint(A, ub=b_ub)
    else:
        constraint = LinearConstraint(np.zeros((0, m)), ub=np.zeros(0))

    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(m)
    
    # Solve
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if res.success:
        selected_indices = np.where(res.x > 0.5)[0]
        selected = set(candidates[i] for i in selected_indices)
    else:
        # Fallback: Greedy if solver fails
        selected = set()
        remaining = list(range(2, n))
        remaining.sort(key=lambda x: -1.0/(n-x))
        for a in remaining:
            if all(gcd(a, b) == 1 for b in selected):
                selected.add(a)
                
    selected.add(1)
    
    return list(selected)
