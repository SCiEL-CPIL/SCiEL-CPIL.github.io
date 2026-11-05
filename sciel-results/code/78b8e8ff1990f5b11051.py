import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def enumerate_maximal_cliques(adj, n):
        """
        Enumerates all maximal cliques using Bron-Kerbosch with pivot.
        Returns a list of bitmasks, each representing a maximal clique.
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            
            # Pivot selection: choose u in P ∪ X that maximizes |P ∩ N(u)|
            PX = P | X
            if PX:
                max_common = -1
                pivot = -1
                temp = PX
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    common = (P & adj[u]).bit_count()
                    if common > max_common:
                        max_common = common
                        pivot = u
                    temp ^= lsb
            else:
                pivot = -1

            if pivot != -1:
                candidates = P & ~(adj[pivot])
            else:
                candidates = P

            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                bk(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb

        bk(0, (1 << n) - 1, 0)
        return cliques

    def solve_instance(adj, n, deadline):
        cliques = enumerate_maximal_cliques(adj, n)
        
        # Filter cliques of size >= 2 (already done in bk, but safe)
        cliques = [c for c in cliques if c.bit_count() >= 2]
        
        if not cliques:
            return []

        k = len(cliques)
        
        # Try ILP first for exact optimal solution
        try:
            # A[i][v] = 1 if v is in clique i
            A = np.zeros((k, n), dtype=np.float64)
            for i, c in enumerate(cliques):
                for v in range(n):
                    if c & (1 << v):
                        A[i][v] = 1.0
            
            # Objective: minimize sum(x_v)
            c_obj = np.ones(n, dtype=np.float64)
            
            # Constraints: sum_{v in C_i} x_v >= 1
            constraints = LinearConstraint(A, 1, np.inf)
            
            # Integrality: all x_v are integers
            integrality = np.ones(n, dtype=np.int8)
            
            # Bounds: 0 <= x_v <= 1
            bounds = Bounds(0, 1)
            
            res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0: # Success
                # Extract selected vertices
                selected = [i for i in range(n) if res.x[i] > 0.5]
                return sorted(selected)
            else:
                # Fallback to greedy if ILP fails
                pass
        except Exception:
            # Fallback to greedy if ILP fails or is not available
            pass

        # Fallback: Greedy Set Cover
        remaining_cliques = set(cliques)
        selected = set()
        
        # Precompute vertex -> cliques bitmask? Or just list.
        # For speed in greedy, let's keep it simple.
        
        while remaining_cliques and time.time() < deadline:
            best_vertex = -1
            max_hits = 0
            
            for v in range(n):
                if v in selected:
                    continue
                v_mask = 1 << v
                hits = 0
                for c in remaining_cliques:
                    if c & v_mask:
                        hits += 1
                
                if hits > max_hits:
                    max_hits = hits
                    best_vertex = v
            
            if best_vertex == -1 or max_hits == 0:
                break
            
            selected.add(best_vertex)
            best_mask = 1 << best_vertex
            
            new_remaining = set()
            for c in remaining_cliques:
                if not (c & best_mask):
                    new_remaining.add(c)
            remaining_cliques = new_remaining
            
            if not remaining_cliques:
                break
        
        if remaining_cliques:
            # Force cover remaining
            for c in remaining_cliques:
                lsb = c & -c
                v = lsb.bit_length() - 1
                selected.add(v)
        
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices (valid but large)
            results.append(list(range(n)))
            continue

        adj = random_graph(n, p, seed)
        transversal = solve_instance(adj, n, deadline)
        results.append(transversal)
        
    return results
