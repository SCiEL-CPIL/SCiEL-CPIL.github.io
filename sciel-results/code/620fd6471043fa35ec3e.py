import time
from random import Random
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill
    
    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(graph, n):
        # graph is a list of edges
        # Build adjacency sets for speed
        adj = [set() for _ in range(n)]
        for u, v in graph:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose u in P ∪ X with max degree in P
            pivot_set = P.union(X)
            if not pivot_set:
                return
            # Heuristic: pick vertex with max neighbors in P
            pivot = max(pivot_set, key=lambda v: len(adj[v].intersection(P)))
            
            # Iterate over P \ N(pivot)
            candidates = P.difference(adj[pivot])
            
            for v in list(candidates):
                # R' = R ∪ {v}
                R_new = R.union([v])
                # P' = P ∩ N(v)
                P_new = P.intersection(adj[v])
                # X' = X ∩ N(v)
                X_new = X.intersection(adj[v])
                
                bk(R_new, P_new, X_new)
                
                # Move v from P to X
                P.discard(v)
                X.add(v)
                
                # Check time
                if time.time() > deadline:
                    return

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        if time.time() > deadline:
            return list(range(n)) # Fallback: return all vertices (valid but large)
            
        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch(edges, n)
        
        if not cliques:
            return []
            
        # Build ILP
        # Variables: x_0 ... x_{n-1}
        # Objective: min sum(x_i)
        c = [1.0] * n
        
        # Constraints: for each clique C, sum_{v in C} x_v >= 1
        # Which is -sum_{v in C} x_v <= -1
        M = len(cliques)
        if M == 0:
            return []
            
        # Sparse matrix A_ub of shape (M, n)
        rows = []
        cols = []
        vals = []
        
        for i, C in enumerate(cliques):
            for v in C:
                rows.append(i)
                cols.append(v)
                vals.append(-1.0)
                
        A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(M, n))
        b_ub = [-1.0] * M
        
        constraints = LinearConstraint(A_ub, -sp.inf, b_ub)
        bounds = Bounds([0.0] * n, [1.0] * n)
        integrality = [1] * n
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.success:
                # res.x is the solution
                # Convert to list of indices where x_i > 0.5
                solution = [i for i in range(n) if res.x[i] > 0.5]
                return sorted(solution)
            else:
                # Fallback to greedy if ILP fails
                pass
        except Exception:
            pass
            
        # Greedy Fallback
        # Sort cliques by size? Or just pick vertices covering most uncovered cliques.
        # Simple greedy:
        # 1. List of uncovered cliques
        # 2. While uncovered:
        #    Find vertex that covers most uncovered cliques
        #    Add to solution
        #    Mark covered cliques as covered
        
        uncovered = list(cliques)
        solution = set()
        
        while uncovered:
            if time.time() > deadline:
                break
            # Count coverage for each vertex
            # Optimize: only consider vertices in uncovered cliques
            counts = {}
            for C in uncovered:
                for v in C:
                    counts[v] = counts.get(v, 0) + 1
            
            if not counts:
                break
                
            # Pick best vertex
            best_v = max(counts, key=counts.get)
            solution.add(best_v)
            
            # Remove covered cliques
            new_uncovered = []
            for C in uncovered:
                if best_v not in C:
                    new_uncovered.append(C)
            uncovered = new_uncovered
            
        return sorted(list(solution))

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        res = solve_instance(n, p, seed)
        out.append(res)
        
    return out
