"""K2,3-free graph construction on 20 vertices using MILP + local search."""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n = 20
    vertices = list(range(n))
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_idx = {e: i for i, e in enumerate(all_edges)}
    m = len(all_edges)
    
    # All pairs (u, v) for K2,3 constraint: sum of common neighbors <= 2
    # For each pair (u, v), common neighbors w must satisfy: deg(u,w) + deg(v,w) - 2*deg(u,v,w) ... 
    # Actually, common neighbors of u and v are vertices w (w != u, w != v) such that (u,w) and (v,w) are edges.
    # K2,3 constraint: for any pair {u,v}, |N(u) ∩ N(v)| <= 2.
    # This is equivalent to: for any triple (u, v, w) where u != v, u != w, v != w,
    # if we have edges (u,w) and (v,w), we count it toward common neighbors.
    # More precisely: for pair (u,v), common_neighbors = {w : w != u, w != v, edge(u,w)=1, edge(v,w)=1}
    # We need |common_neighbors| <= 2 for all pairs.
    
    # Let's use scipy.optimize.milp for a direct ILP approach
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Variables: x_e for each edge e (binary)
        # Objective: maximize sum x_e
        # Constraints: for each pair (u,v), sum over w != u,v of x_{uw} * x_{vw} <= 2
        # This is a quadratic constraint, not linear. 
        # 
        # Alternative: We can use a different formulation.
        # For each vertex w, let S_w = {u : edge(u,w)=1}. 
        # For any two vertices u,v in S_w, the pair (u,v) has w as a common neighbor.
        # K2,3 constraint: for any pair (u,v), the number of w such that both u and v are in S_w is at most 2.
        # 
        # This is still nonlinear. Let's try a different approach:
        # Use a MILP with additional variables y_{u,v,w} = 1 if u and v both connect to w.
        # y_{u,v,w} <= x_{uw}, y_{u,v,w} <= x_{vw}, y_{u,v,w} >= x_{uw} + x_{vw} - 1
        # And constraint: for each pair (u,v), sum over w != u,v of y_{u,v,w} <= 2
        
        # This has O(n^3) variables which is manageable for n=20 (8000 y-variables + 190 x-variables)
        
        # Variable ordering: first x variables (edges), then y variables (triples)
        num_x = m
        # y variables: for each unordered pair (u,v) with u < v, and each w not in {u,v}
        y_vars = []
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(n):
                    if w != u and w != v:
                        y_vars.append((u, v, w))
        num_y = len(y_vars)
        y_idx = {(u, v, w): num_x + i for i, (u, v, w) in enumerate(y_vars)}
        total_vars = num_x + num_y
        
        # Objective: maximize sum x_e => minimize -sum x_e
        c = np.zeros(total_vars)
        for e in range(m):
            c[e] = -1.0  # minimize negative of objective
        
        # Bounds
        lower_bound = np.zeros(total_vars)
        upper_bound = np.ones(total_vars)
        integrality = np.ones(total_vars)  # all integer (binary)
        
        # Constraints
        # 1) y_{u,v,w} <= x_{uw}
        # 2) y_{u,v,w} <= x_{vw}
        # 3) y_{u,v,w} >= x_{uw} + x_{vw} - 1  =>  x_{uw} + x_{vw} - y_{u,v,w} <= 1
        # 4) For each pair (u,v): sum_{w != u,v} y_{u,v,w} <= 2
        
        # Let's build the constraint matrix
        # This will be large, so let's use a sparse approach or build it row by row
        
        # Count constraints:
        # For each y_var: 3 constraints (y <= x_uw, y <= x_vw, x_uw + x_vw - y <= 1)
        # For each pair (u,v): 1 constraint
        num_constraints = 3 * num_y + (n * (n - 1)) // 2
        
        # Build A matrix
        A = np.zeros((num_constraints, total_vars))
        b_lower = np.full(num_constraints, -np.inf)
        b_upper = np.zeros(num_constraints)
        
        row = 0
        for (u, v, w) in y_vars:
            y_i = y_idx[(u, v, w)]
            # Find x_{uw} and x_{vw}
            uw = (min(u, w), max(u, w))
            vw = (min(v, w), max(v, w))
            x_uw = edge_idx.get(uw)
            x_vw = edge_idx.get(vw)
            
            if x_uw is not None and x_vw is not None:
                # Constraint 1: y - x_uw <= 0
                A[row, y_i] = 1
                A[row, x_uw] = -1
                b_upper[row] = 0
                row += 1
                
                # Constraint 2: y - x_vw <= 0
                A[row, y_i] = 1
                A[row, x_vw] = -1
                b_upper[row] = 0
                row += 1
                
                # Constraint 3: x_uw + x_vw - y <= 1
                A[row, x_uw] = 1
                A[row, x_vw] = 1
                A[row, y_i] = -1
                b_upper[row] = 1
                row += 1
        
        # Constraint 4: for each pair (u,v), sum y_{u,v,w} <= 2
        for u in range(n):
            for v in range(u + 1, n):
                for (uu, vv, ww) in y_vars:
                    if uu == u and vv == v:
                        A[row, y_idx[(u, v, ww)]] = 1
                b_upper[row] = 2
                row += 1
        
        # Solve MILP
        if row < num_constraints:
            A = A[:row]
            b_lower = b_lower[:row]
            b_upper = b_upper[:row]
        
        cons = LinearConstraint(A, b_lower, b_upper)
        bounds = Bounds(lower_bound, upper_bound)
        
        result = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if result.status == 0:
            # Extract solution
            x_solution = result.x[:num_x]
            edges = []
            for e, (u, v) in enumerate(all_edges):
                if x_solution[e] > 0.5:
                    edges.append([u, v])
            
            # Verify K2,3-free
            adj = [0] * n
            for u, v in edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            def is_k23_free(adj):
                for u in range(n):
                    for v in range(u + 1, n):
                        cn = adj[u] & adj[v]
                        if bin(cn).count('1') >= 3:
                            return False
                return True
            
            if is_k23_free(adj):
                return edges
            
        # If MILP failed or produced invalid result, fall back to heuristic
    except Exception as e:
        pass
    
    # Fallback: greedy + local search
    adj = [0] * n
    
    def can_add_edge(adj, u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
        if bin(adj[u] & adj[v]).count('1') >= 3:
            return False
        for y in range(n):
            if y == u or y == v:
                continue
            if (adj[u] >> y) & 1:
                if bin(adj[v] & adj[y]).count('1') >= 2:
                    return False
        for x in range(n):
            if x == u or x == v:
                continue
            if (adj[v] >> x) & 1:
                if bin(adj[u] & adj[x]).count('1') >= 2:
                    return False
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def edge_count(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    # Greedy build
    rng = random.Random(42)
    edges_list = all_edges[:]
    rng.shuffle(edges_list)
    for u, v in edges_list:
        if can_add_edge(adj, u, v):
            add_edge(adj, u, v)
    
    # Local search: try to add any missing edge
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
                    break
            if improved:
                break
    
    # Convert to edges list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
