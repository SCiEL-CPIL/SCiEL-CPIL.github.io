import sys
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    
    def bron_kerbosch_maximal_cliques(adj, n):
        """Yield all maximal cliques using Bron-Kerbosch with pivot selection."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            union_px = P | X
            if union_px:
                pivot = max(union_px, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                P.discard(v)
                X.add(v)
                
                bk(R, P_new, X_new)
                
                P.add(v)
                X.discard(v)
                R.discard(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance_ilp(cliques, n):
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        if not relevant_cliques:
            return []
        
        num_cliques = len(relevant_cliques)
        
        # Objective: minimize sum(x_v)
        c = np.ones(n)
        
        # Constraints: sum(x_v for v in C) >= 1 for each clique C
        # We can build the constraint matrix A
        # A is num_cliques x n
        # A[i, v] = 1 if v in relevant_cliques[i], else 0
        
        A = np.zeros((num_cliques, n))
        for i, clique in enumerate(relevant_cliques):
            for v in clique:
                A[i, v] = 1.0
        
        # Constraints: A @ x >= 1
        # LinearConstraint(A, lb=np.ones(num_cliques), ub=np.inf)
        
        constraints = [LinearConstraint(A, lb=np.ones(num_cliques), ub=np.inf)]
        
        # Bounds: 0 <= x_v <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: all variables are integers (binary)
        integrality = np.ones(n)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # res.x contains the solution
            # Since we want a set of vertices, we pick those with x_v > 0.5
            selected = [i for i in range(n) if res.x[i] > 0.5]
            return sorted(selected)
        else:
            # Fallback if ILP fails (should not happen for these sizes)
            # Use a simple greedy fallback
            S = set()
            uncovered = set(range(num_cliques))
            v_to_c = defaultdict(list)
            for i, c_clique in enumerate(relevant_cliques):
                for v in c_clique:
                    v_to_c[v].append(i)
            
            while uncovered:
                best_v = -1
                best_score = -1
                for v in range(n):
                    if v in S:
                        continue
                    cnt = sum(1 for i in v_to_c[v] if i in uncovered)
                    if cnt > best_score:
                        best_score = cnt
                        best_v = v
                if best_v == -1:
                    break
                S.add(best_v)
                for i in v_to_c[best_v]:
                    uncovered.discard(i)
            return sorted(S)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        
        # Solve
        cover = solve_instance_ilp(cliques, n)
        
        out.append(cover)
    
    return out
