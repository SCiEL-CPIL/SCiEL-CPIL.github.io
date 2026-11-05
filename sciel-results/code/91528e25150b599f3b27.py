"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    """
    Find minimum vertex sets meeting every maximal clique (size >= 2) for three random graphs.
    Uses scipy.optimize.milp for exact ILP solution.
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        HAVE_MILP = True
    except ImportError:
        HAVE_MILP = False
    
    def generate_graph(n, p, seed):
        """Generate random graph using same RNG as verifier."""
        from random import Random
        rng = Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return adj, edges
    
    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques using Bron-Kerbosch with pivot.
        Returns list of frozensets of vertices.
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X with max degree in P
            if P or X:
                u = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                return
            
            for v in list(P - adj[u]):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques
    
    def solve_clique_transversal(n, adj, cliques):
        """
        Solve minimum clique transversal: find minimum set of vertices
        that intersects every maximal clique of size >= 2.
        
        ILP formulation:
        - x[v] in {0,1} for each vertex v
        - Minimize sum(x[v])
        - Subject to: for each clique C, sum_{v in C} x[v] >= 1
        """
        if not cliques:
            return []
        
        if HAVE_MILP:
            # Build constraint matrix
            m = len(cliques)
            # Filter out cliques of size 1 (task says "at least two vertices")
            cliques = [c for c in cliques if len(c) >= 2]
            m = len(cliques)
            
            if m == 0:
                return []
            
            # Constraint matrix A: m x n, A[i][j] = 1 if j in cliques[i]
            A = np.zeros((m, n), dtype=np.float64)
            for i, clique in enumerate(cliques):
                for v in clique:
                    A[i, v] = 1.0
            
            # Objective: minimize sum x
            c = np.ones(n, dtype=np.float64)
            
            # Constraints: A @ x >= 1  =>  -A @ x <= -1
            lower = -np.ones(m, dtype=np.float64)
            upper = np.zeros(m, dtype=np.float64)
            
            # Variable bounds: 0 <= x <= 1, integer
            var_bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(n, dtype=np.int8)
            
            constraints = LinearConstraint(A, lb=-np.ones(m), ub=np.zeros(m))
            
            result = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=var_bounds
            )
            
            if result.status == 0:  # Optimal
                solution = [i for i in range(n) if result.x[i] > 0.5]
                return sorted(solution)
            else:
                # Fallback: greedy approach
                return greedy_clique_transversal(n, adj, cliques)
        else:
            return greedy_clique_transversal(n, adj, cliques)
    
    def greedy_clique_transversal(n, adj, cliques):
        """
        Greedy heuristic for clique transversal.
        Repeatedly pick the vertex that covers the most uncovered cliques.
        """
        if not cliques:
            return []
        
        cliques = [c for c in cliques if len(c) >= 2]
        if not cliques:
            return []
        
        uncovered = set(range(len(cliques)))
        selected = set()
        
        while uncovered:
            best_v = None
            best_count = -1
            for v in range(n):
                count = sum(1 for ci in uncovered if v in cliques[ci])
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count <= 0:
                break
            
            selected.add(best_v)
            # Remove all cliques containing best_v
            uncovered = {ci for ci in uncovered if best_v not in cliques[ci]}
        
        return sorted(selected)
    
    # Solve for each instance
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE - 5:
            # Not enough time, use greedy for remaining
            adj, _ = generate_graph(n, p, seed)
            cliques = bron_kerbosch(adj, n)
            results.append(greedy_clique_transversal(n, adj, cliques))
            continue
        
        adj, _ = generate_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        if time.time() > DEADLINE - 10:
            # Tight on time, use greedy
            results.append(greedy_clique_transversal(n, adj, cliques))
            continue
        
        solution = solve_clique_transversal(n, adj, cliques)
        results.append(solution)
    
    return results

# EVOLVE-BLOCK-END
