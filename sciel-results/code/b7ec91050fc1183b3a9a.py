"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm, gcd
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    timeout = 50.0  # Safety margin for 60s limit
    
    # Vertices are 2 to 256
    nodes = list(range(2, 257))
    n = len(nodes)
    
    # Weights: reciprocal of the node value
    weights = np.array([1.0 / x for x in nodes])
    
    # Build conflict constraints
    # Edge (i, j) exists if lcm(nodes[i], nodes[j]) <= 256
    # We need x_i + x_j <= 1 for all such pairs.
    
    # To build the A matrix for LinearConstraint efficiently, 
    # we can count edges or just build a sparse matrix.
    # Given n=255, a dense matrix of 255x255 is 65k floats, which is small.
    # However, scipy.sparse is better practice.
    
    import scipy.sparse as sp
    
    row_indices = []
    col_indices = []
    data = []
    
    for i in range(n):
        a = nodes[i]
        for j in range(i + 1, n):
            b = nodes[j]
            if lcm(a, b) <= 256:
                # Conflict: cannot pick both i and j
                row_indices.append(len(row_indices))
                col_indices.append(i)
                data.append(1)
                
                row_indices.append(len(row_indices))
                col_indices.append(j)
                data.append(1)
                
    if not row_indices:
        # No conflicts, pick all nodes? 
        # But wait, if no conflicts, we pick all.
        # However, lcm(a,b) > 256 is the requirement for NO edge.
        # If no edges, all pairs are compatible.
        return nodes
    
    # Create sparse matrix for constraints: A @ x <= b
    # Each row corresponds to a constraint x_i + x_j <= 1
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(len(row_indices), n))
    b_ub = np.ones(len(row_indices))
    
    # Variables are binary
    integrality = np.ones(n)
    
    # Objective: Maximize sum(weights * x)
    # milp minimizes, so we minimize -sum(weights * x)
    c = -weights
    
    # Bounds
    lb = np.zeros(n)
    ub = np.ones(n)
    bounds = Bounds(lb, ub)
    
    # Constraints
    constraints = LinearConstraint(A, ub=b_ub)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            # res.x contains the solution
            selected_indices = [i for i in range(n) if res.x[i] > 0.5]
            if selected_indices:
                return [nodes[i] for i in selected_indices]
            
            # If solver returned empty (shouldn't happen for non-empty problem), 
            # fallback to best single node
            best_idx = np.argmax(weights)
            return [nodes[best_idx]]
        else:
            # Solver failed or timed out
            pass
            
    except Exception:
        pass
    
    # Fallback: Greedy heuristic or single best node
    # Since we must return a valid subset, the safest is the single node with max weight
    # which is 2 (weight 0.5)
    # But we can do a quick greedy to be better
    # Greedy: sort by weight desc, add if compatible with current set
    current_set = []
    for i in range(n):
        if all(lcm(nodes[i], nodes[j]) > 256 for j in current_set):
            current_set.append(i)
    
    if current_set:
        return [nodes[i] for i in current_set]
        
    return [nodes[0]]

# EVOLVE-BLOCK-END
