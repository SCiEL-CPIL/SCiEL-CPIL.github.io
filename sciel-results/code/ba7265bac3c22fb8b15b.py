import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    def random_graph(n, p, seed):
        """Generates the adjacency matrix for the random graph."""
        import random
        rng = random.Random(seed)
        # Create adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u, v] = 1
                    adj[v, u] = 1
        return adj

    def bron_kerbosch(adj, n):
        """
        Finds all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting.
        Returns a list of lists, where each inner list is a clique (set of vertices).
        """
        cliques = []
        
        def bk(R, P, X):
            # Pruning: if P and X are empty, R is a maximal clique
            if len(P) == 0 and len(X) == 0:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            
            # Pivoting: choose pivot u in P U X that maximizes |P ∩ N(u)|
            # If P is empty, we can't pick a pivot from P, but we handle the base case above.
            # So P is not empty here.
            # We want to maximize |P ∩ N(u)|.
            # We can iterate through candidates in P U X.
            # For efficiency, we can just pick the first one or the one with max degree in P.
            # Let's pick u from P U X.
            candidates = list(P.union(X))
            if not candidates:
                return
            
            # Find pivot with max |P ∩ N(u)|
            max_inter = -1
            pivot = candidates[0]
            for u in candidates:
                # N(u) is neighbors of u
                # P is a set
                # Intersection size
                # We can compute this quickly using bitsets or just sets for n<=40
                # n is small, so set operations are fine.
                inter_size = len(P.intersection(N[u]))
                if inter_size > max_inter:
                    max_inter = inter_size
                    pivot = u
            
            # Recurse on P \ N(pivot)
            # The standard Bron-Kerbosch with pivoting iterates over vertices in P \ N(pivot)
            # Wait, the standard algorithm iterates over v in P \ N(pivot).
            # Let's verify.
            # Yes, Bron-Kerbosch(P, X, R) with pivot u:
            # For each v in P \ N(u):
            #   BK(R ∪ {v}, P ∩ N(v), X ∩ N(v))
            #   P = P \ {v}
            #   X = X ∪ {v}
            
            # We need the set of vertices in P that are NOT neighbors of pivot.
            # P \ N(pivot)
            non_neighbors = P.difference(N[pivot])
            
            # We must iterate over a copy of non_neighbors because we modify P and X
            for v in list(non_neighbors):
                # Recurse
                R_new = R.union({v})
                P_new = P.intersection(N[v])
                X_new = X.intersection(N[v])
                
                bk(R_new, P_new, X_new)
                
                # Update P and X for the next iteration
                P.discard(v)
                X.add(v)

        # Precompute neighbors for each vertex as sets
        N = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i, j] == 1:
                    N[i].add(j)
                    
        # Initial call
        # R = empty, P = all vertices, X = empty
        P_init = set(range(n))
        X_init = set()
        bk(set(), P_init, X_init)
        
        return cliques

    def solve_instance(n, p, seed):
        if time.time() > deadline - 5:
             return list(range(n)) # Fallback if running out of time
            
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        # If no cliques of size >= 2, empty set is the solution
        if not cliques:
            return []
        
        m = len(cliques)
        
        # ILP Formulation
        # Variables: x_i for i in 0..n-1
        # Objective: min sum(x_i)
        c = np.ones(n)
        
        # Constraints: sum_{v in C} x_v >= 1 for each clique C
        # This is equivalent to -sum_{v in C} x_v <= -1
        # Or we can use LinearConstraint with lb=1, ub=inf
        A = np.zeros((m, n))
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i, v] = 1
        
        # Constraints: A @ x >= 1
        # scipy.optimize.milp uses LinearConstraint(A, lb, ub)
        # So we want A @ x >= 1, which is A @ x <= inf and A @ x >= 1
        # LinearConstraint(A, lb=np.ones(m), ub=np.inf)
        
        cons = LinearConstraint(A, lb=np.ones(m), ub=np.inf)
        integrality = np.ones(n)  # All variables are integer
        bounds = Bounds(lb=0, ub=1) # Binary variables
        
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            # res.x contains the optimal values
            # Round to be safe, though they should be 0 or 1
            x = np.round(res.x)
            return [i for i in range(n) if x[i] > 0.5]
        else:
            # Fallback: return all vertices (valid but suboptimal)
            return list(range(n))

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
             # Not enough time, use a quick heuristic or fallback for remaining
             # Fallback: all vertices
             out.append(list(range(n)))
             continue
             
        res_list = solve_instance(n, p, seed)
        out.append(sorted(res_list))
        
    return out
