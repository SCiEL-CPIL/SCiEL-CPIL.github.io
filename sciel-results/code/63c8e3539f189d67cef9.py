import time
import random
from itertools import chain
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    def get_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(adj, n):
        """
        Enumerates all maximal cliques in the graph.
        Returns a list of frozensets of vertices.
        """
        cliques = []
        
        # Pre-compute neighbor sets for faster intersection
        # adj is a list of sets
        R = set()
        P = set(range(n))
        X = set()
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2: # Only consider cliques of size >= 2
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose u in P ∪ X with max degree in P
            u = None
            max_degree = -1
            for v in P.union(X):
                deg = len(P.intersection(adj[v]))
                if deg > max_degree:
                    max_degree = deg
                    u = v
            
            # Candidates are in P \ N(u)
            candidates = P.difference(adj[u])
            
            for v in list(candidates):
                new_R = R.union({v})
                new_P = P.intersection(adj[v])
                new_X = X.intersection(adj[v])
                expand(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)
        
        expand(R, P, X)
        return cliques

    def solve_instance(n, p, seed):
        edges = get_edges(n, p, seed)
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques of size >= 2
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        
        # If no cliques of size >= 2, any empty set is a valid hitting set.
        # But usually, if there are edges, there are cliques.
        if not cliques:
            return []
        
        # Formulate ILP
        # Minimize sum(x_v)
        # s.t. sum(x_v for v in C) >= 1 for each clique C
        #      x_v in {0, 1}
        
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except ImportError:
            # Fallback to greedy if SciPy is not available
            return greedy_hit_set(n, cliques)
        
        # Objective coefficients
        c = np.ones(n)
        
        # Constraints: A @ x >= 1
        # We need to build the constraint matrix A where A[i, v] = 1 if v in clique i
        num_cliques = len(cliques)
        # For efficiency, we can represent constraints directly
        # SciPy milp expects LinearConstraint(A, lb, ub)
        # We can use a sparse matrix or a dense matrix. n is small (<=40), num_cliques might be large but manageable.
        
        # Let's build the matrix
        A = np.zeros((num_cliques, n))
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i, v] = 1.0
        
        # Constraint: A @ x >= 1
        # This is equivalent to -A @ x <= -1
        # Or we can use lb=1, ub=inf for A @ x
        lb = np.ones(num_cliques)
        ub = np.full(num_cliques, np.inf)
        
        constraints = LinearConstraint(A, lb, ub)
        
        # Bounds for x: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        
        # Integrate variables
        integrality = np.ones(n)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Success
            # Extract solution
            solution = res.x
            # Round to nearest integer (should be close to 0 or 1)
            vertices = [i for i in range(n) if solution[i] > 0.5]
            return sorted(vertices)
        else:
            # Fallback if ILP fails
            return greedy_hit_set(n, cliques)

    def greedy_hit_set(n, cliques):
        """
        Greedy algorithm for hitting set.
        Iteratively picks the vertex that hits the most remaining cliques.
        """
        remaining_cliques = [set(c) for c in cliques]
        if not remaining_cliques:
            return []
            
        # Map vertex to list of cliques it belongs to
        # For efficiency, we can just iterate
        selected = []
        
        # Pre-compute frequency of each vertex in cliques
        # But cliques change as we remove them.
        
        # Simple greedy:
        # 1. Count how many remaining cliques each vertex hits.
        # 2. Pick the vertex with max count.
        # 3. Remove all cliques hit by this vertex.
        # 4. Repeat.
        
        active_cliques = [set(c) for c in cliques]
        
        while active_cliques:
            # Check time
            if time.time() > deadline - 5.0:
                # Return current selection + all vertices in remaining cliques to ensure validity
                # This is a valid, though not minimal, completion
                extra = set()
                for c in active_cliques:
                    extra.update(c)
                return sorted(list(set(selected) | extra))
                
            counts = {}
            for c in active_cliques:
                for v in c:
                    counts[v] = counts.get(v, 0) + 1
            
            # Find vertex with max count
            best_v = None
            max_count = -1
            for v, cnt in counts.items():
                if cnt > max_count:
                    max_count = cnt
                    best_v = v
            
            if best_v is None:
                break
                
            selected.append(best_v)
            
            # Remove all cliques containing best_v
            new_active = []
            for c in active_cliques:
                if best_v not in c:
                    new_active.append(c)
            active_cliques = new_active
            
        return sorted(selected)

    results = []
    params = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in params:
        if time.time() > deadline - 10.0:
            # Not enough time for ILP, use seed solution (all non-isolated vertices)
            edges = get_edges(n, p, seed)
            touched = set()
            for u, v in edges:
                touched.add(u)
                touched.add(v)
            results.append(sorted(touched))
        else:
            sol = solve_instance(n, p, seed)
            results.append(sol)
            
    return results
# EVOLVE-BLOCK-END
