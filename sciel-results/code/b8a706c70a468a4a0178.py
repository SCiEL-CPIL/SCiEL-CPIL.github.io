import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    # 2. Filter vertices: 2*v must not be a square
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1]
        
    # Map vertex value to index
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 3. Build adjacency matrix constraints
    # We want to maximize sum(x_i) subject to x_i + x_j <= 1 for all edges (i,j)
    # where edge exists if valid_vertices[i] + valid_vertices[j] is a square.
    
    # Number of constraints is the number of edges
    # We can build the constraint matrix A_ub and b_ub
    # Or use LinearConstraint with sparse matrices for efficiency
    
    from scipy.sparse import lil_matrix
    
    # Estimate max edges: worst case complete graph n^2/2 = 256^2/2 ~ 32768. 
    # Actual edges are much fewer.
    
    # Let's collect edges first
    edges = []
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                edges.append((i, j))
                
    n_edges = len(edges)
    
    if n_edges == 0:
        return valid_vertices[:] # All vertices are independent
        
    # Build A matrix (n_edges x n_vars)
    # A[k, i] = 1 if edge k involves vertex i
    # A[k, j] = 1 if edge k involves vertex j
    
    A = lil_matrix((n_edges, n_vars))
    for k, (i, j) in enumerate(edges):
        A[k, i] = 1
        A[k, j] = 1
        
    A = A.tocsr()
    
    # Objective: maximize sum(x) -> minimize -sum(x)
    c = -np.ones(n_vars)
    
    # Constraints:
    # 1. x_i + x_j <= 1 for all edges
    # 2. 0 <= x_i <= 1 (integrality handled by milp)
    
    constraints = [
        LinearConstraint(A, ub=np.ones(n_edges)),
        LinearConstraint(np.eye(n_vars), lb=0, ub=1)
    ]
    
    integrality = np.ones(n_vars) # Integer variables
    
    bounds = Bounds(lb=0, ub=1)
    
    try:
        # Use time limit in solver options if possible, or rely on deadline check
        # scipy milp doesn't have a direct wall-clock time limit option in all versions, 
        # but we can check time before/after. 
        # For small N, it should be fast.
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            # Round to nearest integer to be safe
            x = np.round(x)
            
            selected_vertices = []
            for i in range(n_vars):
                if x[i] > 0.5:
                    selected_vertices.append(valid_vertices[i])
                    
            if selected_vertices:
                return selected_vertices
            else:
                # Fallback if ILP returns empty (should not happen for non-empty graph with valid vertices)
                return [valid_vertices[0]]
        else:
            # ILP failed, fallback to greedy
            pass
            
    except Exception:
        pass
        
    # Fallback: Greedy construction
    # Rebuild neighbors for greedy
    neighbors = {v: set() for v in valid_vertices}
    for i, j in edges:
        vi = valid_vertices[i]
        vj = valid_vertices[j]
        neighbors[vi].add(vj)
        neighbors[vj].add(vi)
        
    # Sort by degree ascending
    vs = sorted(valid_vertices, key=lambda v: len(neighbors[v]))
    
    S = set()
    for v in vs:
        if neighbors[v].isdisjoint(S):
            S.add(v)
            
    return list(S)
