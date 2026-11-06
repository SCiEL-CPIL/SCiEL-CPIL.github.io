import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 55
    
    N = 256
    # Nodes are 2..256, map to indices 0..254
    nodes = list(range(2, N + 1))
    n_vars = len(nodes)
    
    # Weights
    c = -np.array([1.0 / x for x in nodes])  # Minimize negative sum
    
    # Build constraints
    # x_i + x_j <= 1 for all conflicting pairs
    A_rows = []
    # We can build the matrix directly or use sparse
    
    # Precompute conflicts
    # conflict[i] is a list of j > i that conflict with i
    conflicts = [[] for _ in range(n_vars)]
    
    for i_idx, i in enumerate(nodes):
        for j_idx in range(i_idx + 1, n_vars):
            j = nodes[j_idx]
            if i * j <= 256 * gcd(i, j):
                conflicts[i_idx].append(j_idx)
                
    # Create constraint matrix
    # Number of constraints is sum of len(conflicts[i])
    num_constraints = sum(len(c_list) for c_list in conflicts)
    
    if num_constraints == 0:
        # No conflicts, take all
        return nodes
        
    # Build sparse matrix for efficiency
    import scipy.sparse as sp
    
    row_indices = []
    col_indices = []
    data = []
    
    for i_idx, c_list in enumerate(conflicts):
        for j_idx in c_list:
            row_indices.append(len(row_indices)) # This is not efficient, just count
            # Actually, we need to build the matrix properly.
            
    # Let's just build lists for sparse matrix
    rows = []
    cols = []
    vals = []
    
    for i_idx, c_list in enumerate(conflicts):
        for j_idx in c_list:
            rows.append(len(rows)) # Wait, row index is the constraint index
            # We need a unique row index for each constraint.
            # Let's just iterate and append.
            
    # Re-do properly
    rows = []
    cols = []
    vals = []
    constraint_idx = 0
    for i_idx, c_list in enumerate(conflicts):
        for j_idx in c_list:
            rows.append(constraint_idx)
            cols.append(i_idx)
            vals.append(1)
            
            rows.append(constraint_idx)
            cols.append(j_idx)
            vals.append(1)
            
            constraint_idx += 1
            
    A = sp.csr_matrix((vals, (rows, cols)), shape=(num_constraints, n_vars))
    
    # Constraint: A x <= 1
    # milp expects Ax >= lb and Ax <= ub
    lb = np.zeros(num_constraints)
    ub = np.ones(num_constraints)
    
    constraint = LinearConstraint(A, lb, ub)
    
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars)
    
    try:
        res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            x = res.x
            # x should be 0 or 1
            subset = [nodes[i] for i in range(n_vars) if x[i] > 0.5]
            if subset:
                return subset
            else:
                # Fallback if empty (should not happen as singletons are valid)
                return [nodes[0]]
        else:
            # Fallback to greedy if solver fails
            pass
    except Exception:
        pass
    
    # Fallback: Greedy
    # Sort by weight descending (index ascending)
    greedy_set = []
    used = set()
    for i in nodes:
        ok = True
        for y in greedy_set:
            if i * y <= 256 * gcd(i, y):
                ok = False
                break
        if ok:
            greedy_set.append(i)
            
    return greedy_set
