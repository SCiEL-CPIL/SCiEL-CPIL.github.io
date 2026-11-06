import time
import math
import random as _random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the seed program."""
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting.
        """
        cliques = []
        
        def bk(R, P, X):
            if time.time() - start_time > DEADLINE:
                return
            
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u in P ∪ X to maximize |P ∩ adj[u]|
            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            
            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])
            
            for v in candidates:
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Enumerate maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        m = len(cliques)
        
        # If no cliques, return empty.
        # We need to find a set of vertices hitting all cliques.
        
        # Formulate MILP
        # Variables x_v for v in range(n)
        # Minimize sum(x_v)
        # Subject to: for each clique C_i, sum_{v in C_i} x_v >= 1
        # x_v in {0, 1}
        
        c = np.ones(n)
        
        # Build constraint matrix A
        # Rows correspond to cliques, columns to vertices
        # A[i, v] = 1 if v in cliques[i] else 0
        
        # Efficient construction
        A = np.zeros((m, n), dtype=float)
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i, v] = 1.0
        
        # Constraints: A @ x >= 1
        # LinearConstraint(A, lb=np.ones(m), ub=np.inf)
        
        try:
            res = milp(
                c=c,
                constraints=[LinearConstraint(A, lb=np.ones(m), ub=np.inf)],
                integrality=np.ones(n),
                bounds=Bounds(lb=np.zeros(n), ub=np.ones(n))
            )
            
            if res.status == 0:
                # res.x contains the solution
                sol_vertices = [i for i in range(n) if res.x[i] > 0.5]
                return sorted(sol_vertices)
            else:
                # Fallback to greedy if MILP fails (should not happen)
                pass
                
        except Exception as e:
            pass

        # Fallback: Greedy Set Cover
        # This is just a safety net, MILP should work.
        vertex_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(ci)
        
        uncovered = set(range(m))
        selected = set()
        
        while uncovered:
            best_v = -1
            best_cov = 0
            for v in range(n):
                if v in selected:
                    continue
                cov = len(uncovered & vertex_cliques[v])
                if cov > best_cov:
                    best_cov = cov
                    best_v = v
            if best_v == -1:
                break
            selected.add(best_v)
            uncovered -= vertex_cliques[best_v]
            
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)
        
    return results
