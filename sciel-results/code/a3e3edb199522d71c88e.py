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

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                cliques.append(set(R))
                return
            # Choose pivot from P ∪ X to minimize branching
            pivot_set = P | X
            if pivot_set:
                # Pick pivot with max degree in P
                pivot = max(pivot_set, key=lambda v: sum(1 for u in P if adj[v][u]))
                candidates = [v for v in P if not adj[v][pivot]]
            else:
                candidates = list(P)
            
            for v in candidates:
                bk(R | {v}, P & {u for u in P if adj[v][u]}, X & {u for u in X if adj[v][u]})
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def greedy_hitting_set(cliques, n):
        """Greedy hitting set: repeatedly pick vertex hitting most remaining cliques."""
        if not cliques:
            return set()
        
        unhit = list(cliques)
        result = set()
        
        while unhit:
            # Count how many unhit cliques each vertex hits
            counts = {}
            for c in unhit:
                for v in c:
                    counts[v] = counts.get(v, 0) + 1
            
            # Pick vertex with max count
            if not counts:
                break
            best_v = max(counts, key=lambda v: counts[v])
            result.add(best_v)
            
            # Remove all cliques hit by best_v
            unhit = [c for c in unhit if best_v not in c]
        
        return result

    def solve_ilp(cliques, n):
        """Solve the hitting set problem using Integer Linear Programming."""
        if not cliques:
            return set()
        
        # Variables: x_v for v in 0..n-1, binary
        # Objective: minimize sum(x_v)
        c = np.ones(n)
        
        # Constraints: For each clique C, sum_{v in C} x_v >= 1
        m = len(cliques)
        A = np.zeros((m, n))
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i, v] = 1
        
        # Constraint: A x >= 1
        lb = np.ones(m)
        ub = np.full(m, n)  # Upper bound
        
        constraints = LinearConstraint(A, lb, ub)
        
        # Bounds for variables: 0 <= x_v <= 1
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        
        # Integrality: all variables are integer (binary)
        integrality = np.ones(n)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": 30, "mip_rel_gap": 0.0})
        
        if not res.success:
            # Fallback to greedy if ILP fails
            return greedy_hitting_set(cliques, n)
            
        # res.x contains the optimal values (0.0 or 1.0)
        hitting_set = set()
        for v in range(n):
            if res.x[v] > 0.5:
                hitting_set.add(v)
                
        return hitting_set

    def post_process(hitting_set, cliques, n):
        """Try to remove vertices from hitting_set if still valid."""
        if not hitting_set:
            return hitting_set
        
        result = set(hitting_set)
        # Try removing each vertex one at a time
        for v in sorted(result):
            candidate = result - {v}
            if not candidate:
                break
            # Check if candidate still hits all cliques
            valid = True
            for clique in cliques:
                if not any(u in candidate for u in clique):
                    valid = False
                    break
            if valid:
                result = candidate
        
        return result

    def local_search_improve(hitting_set, cliques, n):
        """Try to improve hitting set by swapping vertices."""
        if not hitting_set:
            return hitting_set
        
        result = set(hitting_set)
        improved = True
        
        while improved:
            improved = False
            for v in list(result):
                # Try removing v
                candidate = result - {v}
                if not candidate:
                    continue
                # Check if candidate still hits all cliques
                valid = True
                for clique in cliques:
                    if not any(u in candidate for u in clique):
                        valid = False
                        break
                if valid:
                    result = candidate
                    improved = True
                    break
        
        return result

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = generate_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        # Filter to cliques of size >= 2
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]
        
        # Use ILP to find optimal hitting set
        hitting_set = solve_ilp(relevant_cliques, n)
        
        # Post-process to try to reduce size
        hitting_set = post_process(hitting_set, relevant_cliques, n)
        
        # Local search improvement
        hitting_set = local_search_improve(hitting_set, relevant_cliques, n)
        
        out.append(sorted(hitting_set))
    
    return out
