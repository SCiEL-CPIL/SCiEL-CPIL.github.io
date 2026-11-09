def construct():
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    import itertools
    import time

    n = 16
    vertices = range(n)
    edges = list(itertools.combinations(vertices, 2))
    m = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    # Precompute triangles
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((edge_index[(u, v)], edge_index[(v, w)], edge_index[(u, w)]))
    
    def is_3_colorable(edge_set):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Backtracking 3-coloring
        color = [-1] * n
        
        # Order vertices by degree descending (DSATUR heuristic simplified)
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try colors 0, 1, 2
            for c in range(3):
                if all(color[nei] != c for nei in adj[v]):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    # ILP Setup
    # Minimize -sum(x_e)
    c = -np.ones(m)
    
    # Triangle constraints: x_uv + x_vw + x_uw <= 2
    # A_ub * x <= b_ub
    A_ub = np.zeros((len(triangles), m))
    b_ub = np.full(len(triangles), 2.0)
    for i, (e1, e2, e3) in enumerate(triangles):
        A_ub[i, e1] = 1
        A_ub[i, e2] = 1
        A_ub[i, e3] = 1
        
    constraints = LinearConstraint(A_ub, ub=b_ub)
    bounds = Bounds(0, 1)
    integrality = np.ones(m)
    
    start_time = time.time()
    time_limit = 110
    
    current_max_edges = m  # Start with upper bound
    best_solution = None
    
    while time.time() - start_time < time_limit:
        # Add constraint: sum(x_e) <= current_max_edges - 1
        # We need to add this to the existing constraints
        # LinearConstraint supports multiple blocks? 
        # It's easier to construct the full A_ub each time or use a list of constraints.
        # scipy.optimize.milp accepts a LinearConstraint object.
        # We can combine the triangle constraints and the sum constraint.
        
        # Create new A_ub with an extra row for the sum constraint
        A_new = np.vstack([A_ub, np.ones(m)])
        b_new = np.append(b_ub, current_max_edges - 1)
        
        constraints_new = LinearConstraint(A_new, ub=b_new)
        
        res = milp(c, constraints=constraints_new, integrality=integrality, bounds=bounds)
        
        if not res.success:
            break
            
        # Extract solution
        x = res.x
        edge_set = set()
        for i, e in enumerate(edges):
            if x[i] > 0.5:
                edge_set.add(e)
        
        num_edges = len(edge_set)
        
        if num_edges == 0:
            break
            
        # Check 3-colorability
        if not is_3_colorable(edge_set):
            best_solution = edge_set
            break
            
        # If 3-colorable, we need to exclude this solution or just lower the bound
        # Lowering the bound is safe because we want the MAX edges.
        # If the max triangle-free graph is 3-colorable, we need the next best.
        current_max_edges = num_edges
        
    if best_solution is None:
        # Fallback to Grötzsch graph if ILP fails or finds nothing
        # This shouldn't happen as Grötzsch is triangle-free and not 3-colorable
        groetzsch_edges = set()
        for i in range(5):
            groetzsch_edges.add((i, (i+1) % 5))
        for i in range(5):
            u, w = i, (i+1) % 5
            groetzsch_edges.add((u, 5+w))
            groetzsch_edges.add((w, 5+u))
        for i in range(5):
            groetzsch_edges.add((5+i, 10))
        # Add 5 isolated vertices or connect them? 
        # Isolated vertices don't add edges. 
        # We can add edges between 11..15 and 0..10 if triangle-free.
        # Let's just return the Grötzsch edges (mapped to 0..15)
        # Vertices 11-15 are isolated.
        best_solution = groetzsch_edges
        
    return [list(e) for e in best_solution]
