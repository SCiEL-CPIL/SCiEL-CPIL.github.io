def construct():
    import sys
    sys.setrecursionlimit(10000)
    import random
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = adj[v][u] = True
        return adj

    def greedy_vertex_cover(adj, n):
        """Greedy Vertex Cover: repeatedly pick vertex with highest degree."""
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.add((u, v))
        
        if not edges:
            return set()

        cover = set()
        while edges:
            deg = {v: 0 for v in range(n)}
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            
            best_v = max(range(n), key=lambda x: deg[x])
            if deg[best_v] == 0:
                break
                
            cover.add(best_v)
            new_edges = set()
            for u, v in edges:
                if u != best_v and v != best_v:
                    new_edges.add((u, v))
            edges = new_edges
            
        return cover

    def solve_vc_ilp(adj, n):
        """Solve Minimum Vertex Cover using ILP on edges."""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append((u, v))
        
        if not edges:
            return set()

        m = len(edges)
        
        # Variables: x_v for v in 0..n-1, binary
        # Objective: minimize sum(x_v)
        c = np.ones(n)
        
        # Constraints: For each edge (u, v), x_u + x_v >= 1
        A = np.zeros((m, n))
        for i, (u, v) in enumerate(edges):
            A[i, u] = 1
            A[i, v] = 1
            
        lb = np.ones(m)
        ub = np.full(m, 2)
        
        constraints = LinearConstraint(A, lb, ub)
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        integrality = np.ones(n)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if not res.success:
            # Fallback to greedy
            return greedy_vertex_cover(adj, n)
            
        cover = set()
        for v in range(n):
            if res.x[v] > 0.5:
                cover.add(v)
                
        return cover

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = generate_graph(n, p, seed)
        
        # Solve Minimum Vertex Cover
        vc = solve_vc_ilp(adj, n)
        
        out.append(sorted(vc))
    
    return out
