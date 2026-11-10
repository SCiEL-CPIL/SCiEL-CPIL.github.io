import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for I vertices to C vertices
        adj_I = [set() for _ in range(n)]
        C_edges = set()
        
        for u, v in edges:
            if u < k and v < k:
                C_edges.add((u, v))
            elif u >= k and v < k:
                adj_I[u].add(v)
                adj_I[v].add(u)
            elif u < k and v >= k:
                adj_I[v].add(u)
                adj_I[u].add(v)

        I_vertices = list(range(k, n))
        C_vertices = list(range(k))
        
        result_cliques = []
        covered_C_edges = set()
        
        # Strategy: Use singletons for I vertices.
        # For each u in I, for each v in N(u), create clique [u, v].
        # This covers all I-C edges.
        # It covers NO C-C edges (since clique size 2, only edge is (u,v)).
        for u in I_vertices:
            for v in adj_I[u]:
                result_cliques.append([u, v])
        
        # Now, cover remaining C-C edges.
        # All C-C edges are uncovered.
        # We need to cover C_edges with cliques in C.
        # C is a clique, so any subset is a clique.
        # We want to minimize number of cliques.
        # This is equivalent to covering the edges of the graph (C_vertices, C_edges) with cliques.
        # Since C is complete, the graph is (C_vertices, C_edges).
        # Wait, C_edges is the set of ALL edges in C?
        # Yes, because C is a clique in the split graph definition.
        # "a clique on the first k vertices"
        # So C_edges = all pairs in C_vertices.
        
        # If C_edges is the complete graph on k vertices, we can cover it with 1 clique: C_vertices.
        # Is that allowed?
        # "Every listed set must be a clique". Yes.
        # "Every edge must lie in exactly one listed clique".
        # If we add C_vertices as a clique, it covers all C-C edges.
        # Do any C-C edges lie in the [u, v] cliques?
        # No, [u, v] only contains edge (u, v) which is I-C.
        # So we can just add C_vertices.
        
        if C_vertices:
            result_cliques.append(C_vertices)
            
        # Convert to lists of ints
        final_result = []
        for clique in result_cliques:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)
        
        return final_result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            trivial = [[int(u), int(v)] for u, v in edges]
            results.append(trivial)
        else:
            res = solve_instance(*spec)
            results.append(res)
            
    return results
