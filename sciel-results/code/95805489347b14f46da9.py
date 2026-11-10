import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    N = 256
    # 1. Identify invalid nodes (self-loops)
    invalid_nodes = set()
    for k in range(2, 23): # k^2 up to 22^2=484
        s = k * k
        if s % 2 == 0:
            a = s // 2
            if 1 <= a <= N:
                invalid_nodes.add(a)
    
    valid_nodes = [i for i in range(1, N + 1) if i not in invalid_nodes]
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}
    n_valid = len(valid_nodes)
    
    # 2. Build edges
    edges = []
    squares = [k * k for k in range(2, 23)] # 4 to 484
    
    for u in valid_nodes:
        idx_u = node_to_idx[u]
        for s in squares:
            v = s - u
            if v == u:
                continue # Handled by invalid_nodes
            if v > u and v in node_to_idx:
                idx_v = node_to_idx[v]
                edges.append((idx_u, idx_v))
    
    # 3. Formulate MILP
    # Maximize sum(x) subject to x_u + x_v <= 1 for all edges
    # Minimize -sum(x)
    c = -np.ones(n_valid)
    
    if not edges:
        # No edges, take all valid nodes
        return valid_nodes
    
    # Constraint matrix A
    # Rows: edges, Cols: nodes
    A_rows = []
    A_cols = []
    A_data = []
    
    for i, (u, v) in enumerate(edges):
        A_rows.append(i)
        A_cols.append(u)
        A_data.append(1.0)
        
        A_rows.append(i)
        A_cols.append(v)
        A_data.append(1.0)
        
    A = csr_matrix((A_data, (A_rows, A_cols)), shape=(len(edges), n_valid))
    
    # Constraints: A x <= 1
    constraints = LinearConstraint(A, ub=np.ones(len(edges)))
    
    # Bounds: 0 <= x <= 1
    bounds = [(0, 1) for _ in range(n_valid)]
    
    # Integrality: all integer
    integrality = np.ones(n_valid)
    
    # Solve
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={'time_limit': 60})
        
        if res.success and res.x is not None:
            # res.x is the solution vector
            selected = [valid_nodes[i] for i in range(n_valid) if res.x[i] > 0.5]
            # Verify validity just in case
            if _is_valid(selected):
                return selected
            else:
                # Fallback if something went wrong with interpretation
                return _greedy_fallback(valid_nodes, node_to_idx, edges)
        else:
            # Solver failed or timed out without a good solution?
            # Try to extract incumbent if available? 
            # milp might not expose incumbent directly if not successful.
            # Fallback to greedy
            return _greedy_fallback(valid_nodes, node_to_idx, edges)
            
    except Exception as e:
        # Fallback to greedy
        return _greedy_fallback(valid_nodes, node_to_idx, edges)

def _is_valid(subset):
    if not subset:
        return False
    s_set = set(subset)
    squares = set(k*k for k in range(1, 23))
    for a in s_set:
        if 2*a in squares:
            return False
    for i, a in enumerate(s_set):
        for b in list(s_set)[i+1:]:
            if a + b in squares:
                return False
    return True

def _greedy_fallback(valid_nodes, node_to_idx, edges):
    # Build adjacency list
    adj = {i: set() for i in range(len(valid_nodes))}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Sort by degree
    sorted_nodes = sorted(range(len(valid_nodes)), key=lambda i: len(adj[i]))
    
    selected = []
    selected_set = set()
    
    for i in sorted_nodes:
        if i in selected_set:
            continue
        # Check if any neighbor is selected
        if any(n in selected_set for n in adj[i]):
            continue
        selected.append(valid_nodes[i])
        selected_set.add(i)
        
    return selected
