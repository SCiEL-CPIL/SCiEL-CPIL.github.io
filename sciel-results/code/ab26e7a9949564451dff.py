"""Rewritten program: maximize edge-disjoint Hamiltonian cycles on the 21-vertex host using simultaneous ILP."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _find_k_hamiltonian_cycles(n, edges, K):
    """
    Find up to K edge-disjoint Hamiltonian cycles using a single ILP.
    Returns a list of cycles.
    """
    m = len(edges)
    arc_list = []
    for (a, b) in edges:
        arc_list.append((a, b))
        arc_list.append((b, a))
    A = len(arc_list)
    
    # Variables:
    # x_{e,j} for e in [0, m), j in [0, K)
    # y_{arc,j} for arc in [0, A), j in [0, K)
    # w_{v,j} for v in [0, n), j in [0, K)
    
    n_x = m * K
    n_y = A * K
    n_w = n * K
    nvar = n_x + n_y + n_w
    
    # Objective: Maximize total edges used (sum x)
    c = np.zeros(nvar)
    for j in range(K):
        for i in range(m):
            c[j*m + i] = -1.0
            
    integrality = np.zeros(nvar)
    integrality[:n_x] = 1
    integrality[n_x:n_x+n_y] = 1
    integrality[n_x+n_y:] = 1
    
    lb = np.zeros(nvar)
    ub = np.ones(nvar)
    ub[n_x+n_y:] = n  # w in [1, n]
    
    cons = []
    
    # Helper to get variable index
    def idx_x(e, j):
        return j * m + e
        
    def idx_y(arc, j):
        return n_x + j * A + arc
        
    def idx_w(v, j):
        return n_x + n_y + j * n + v
        
    # 1. Degree constraints: for each v, j, sum x_{e,j} for e incident to v = 2
    # Precompute incident edges for each vertex
    incident = [[] for _ in range(n)]
    for i, (a, b) in enumerate(edges):
        incident[a].append(i)
        incident[b].append(i)
        
    for j in range(K):
        for v in range(n):
            row = np.zeros(nvar)
            for e in incident[v]:
                row[idx_x(e, j)] = 1.0
            cons.append(LinearConstraint(row, 2, 2))
            
    # 2. Flow conservation: for each v != 0, j, sum_in y - sum_out y = 0
    # Precompute arcs in/out for each vertex
    arcs_in = [[] for _ in range(n)]
    arcs_out = [[] for _ in range(n)]
    for k, (u, v) in enumerate(arc_list):
        arcs_in[v].append(k)
        arcs_out[u].append(k)
        
    for j in range(K):
        for v in range(1, n):
            row = np.zeros(nvar)
            for k in arcs_in[v]:
                row[idx_y(k, j)] = 1.0
            for k in arcs_out[v]:
                row[idx_y(k, j)] = -1.0
            cons.append(LinearConstraint(row, 0, 0))
            
    # 3. Link y to x: y_{u,v,j} + y_{v,u,j} <= x_{uv,j}
    # We need mapping from edge to arcs
    # arc_list is built by iterating edges, so for edge i, arcs are 2*i and 2*i+1
    for j in range(K):
        for i in range(m):
            row = np.zeros(nvar)
            row[idx_x(i, j)] = 1.0
            row[idx_y(2*i, j)] = 1.0
            row[idx_y(2*i+1, j)] = 1.0
            cons.append(LinearConstraint(row, 0, 1))
            
    # 4. MTZ constraints: w_u - w_v + 1 <= (n-1)(1 - y_{u,v})
    # w_u - w_v + 1 + (n-1)y_{u,v} <= n-1
    for j in range(K):
        for k, (u, v) in enumerate(arc_list):
            row = np.zeros(nvar)
            row[idx_w(u, j)] = 1.0
            row[idx_w(v, j)] = -1.0
            row[idx_y(k, j)] = n - 1
            cons.append(LinearConstraint(row, -np.inf, n - 2))
            
    # 5. Fix root w_{0,j} = 1
    for j in range(K):
        row = np.zeros(nvar)
        row[idx_w(0, j)] = 1.0
        cons.append(LinearConstraint(row, 1, 1))
        
    # 6. Edge disjointness: sum_j x_{e,j} <= 1
    for i in range(m):
        row = np.zeros(nvar)
        for j in range(K):
            row[idx_x(i, j)] = 1.0
        cons.append(LinearConstraint(row, 0, 1))
        
    res = milp(c=c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub))
    
    if not res.success:
        return []
        
    x = res.x[:n_x]
    
    # Reconstruct cycles
    cycles = []
    for j in range(K):
        # Get selected edges for cycle j
        selected_edges = []
        for i in range(m):
            if x[idx_x(i, j)] > 0.5:
                selected_edges.append(edges[i])
                
        if len(selected_edges) != n:
            continue
            
        # Build adjacency for this cycle
        adj_c = [set() for _ in range(n)]
        for a, b in selected_edges:
            adj_c[a].add(b)
            adj_c[b].add(a)
            
        # Check degrees
        valid = True
        for v in range(n):
            if len(adj_c[v]) != 2:
                valid = False
                break
        if not valid:
            continue
            
        # Walk the cycle
        start = 0
        cyc = [start]
        prev = -1
        cur = start
        for _ in range(n):
            nxts = [u for u in adj_c[cur] if u != prev]
            if not nxts:
                valid = False
                break
            prev, cur = cur, nxts[0]
            if cur == start:
                break
            cyc.append(cur)
            
        if not valid or len(cyc) != n:
            continue
            
        # Verify closing edge
        a, b = cyc[-1], cyc[0]
        if (a, b) not in selected_edges and (b, a) not in selected_edges:
             # Note: selected_edges contains tuples (a,b). 
             # adj_c check already ensures connectivity. 
             # The walk ensures consecutive edges. The last edge is implicit in the 2-degree constraint and walk.
             pass
             
        cycles.append(cyc)
        
    return cycles

def construct():
    n = P['n']
    edges = [tuple(e) for e in P['edges']]
    
    # Try to find up to 3 cycles
    # If 3 fails or returns <3, try 2, then 1
    # Actually, the ILP with K=3 will find the optimal set of size <=3.
    # So we just run K=3.
    
    cycles = _find_k_hamiltonian_cycles(n, edges, 3)
    
    if not cycles:
        # Fallback to greedy if ILP fails completely
        cycles = [list(range(n))]
        
    # Ensure types
    out = [tuple(int(v) for v in c) for c in cycles]
    return out
# EVOLVE-BLOCK-END
