"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import networkx as nx
from scipy.optimize import linear_sum_assignment

def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges + cross_edges)
    
    def solve_instance(n, k, p, seed):
        deadline = time.time() + 110  # safety margin
        
        # Generate edges
        from random import Random
        rng = Random(seed)
        big_clique_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        
        all_edges = big_clique_edges + cross_edges
        
        # Build adjacency for cross edges
        # For each independent vertex v, store which big-clique vertices it connects to
        cross_adj = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            cross_adj[v].add(u)
        
        # Build bipartite matching: left = big_clique_edges, right = independent vertices
        # Connect edge (a,b) to vertex v if a in cross_adj[v] and b in cross_adj[v]
        
        num_big_edges = len(big_clique_edges)
        num_indep = n - k
        
        if num_big_edges == 0 or num_indep == 0:
            # No merging possible
            return [[list(e) for e in all_edges]]
        
        # Build cost matrix for linear_sum_assignment
        # We want to maximize matches, so minimize cost = 0 if matchable, large if not
        
        INF = 10**9
        cost = [[INF] * num_indep for _ in range(num_big_edges)]
        
        for i, (a, b) in enumerate(big_clique_edges):
            for v in range(k, n):
                if a in cross_adj[v] and b in cross_adj[v]:
                    cost[i][v] = 0
        
        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Count valid matches
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost[r][c] == 0:
                matches.append((r, c))
        
        # Build cliques
        cliques = []
        used_big_edges = set()
        used_indep = set()
        
        for r, c in matches:
            a, b = big_clique_edges[r]
            v = k + c
            cliques.append([a, b, v])
            used_big_edges.add(r)
            used_indep.add(v)
        
        # Remaining big clique edges
        for i, (a, b) in enumerate(big_clique_edges):
            if i not in used_big_edges:
                cliques.append([a, b])
        
        # Remaining cross edges
        for u, v in cross_edges:
            # Check if this cross edge is already in a triangle
            # It's in a triangle if v is used and the big-clique edge (u, other) is matched
            # Actually, we need to track which cross edges are covered
            pass
        
        # This is getting complicated. Let me track covered edges properly.
        
        return cliques
    
    # This is getting too complex for a quick implementation. Let me use a simpler approach.
    # Just use the seed approach but try to merge triangles greedily.
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        from random import Random
        rng = Random(seed)
        big_clique_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        
        all_edges = big_clique_edges + cross_edges
        
        # Greedy triangle merging
        # For each independent vertex, find a big-clique edge (a,b) where both (a,v) and (b,v) exist
        # Form triangle, mark edges as used
        
        cross_adj = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            cross_adj[v].add(u)
        
        used_edges = set()
        cliques = []
        
        # Sort independent vertices by degree (descending) for better greedy
        indep_vertices = sorted(range(k, n), key=lambda v: len(cross_adj[v]), reverse=True)
        
        for v in indep_vertices:
            if len(cross_adj[v]) < 2:
                continue
            # Find an unused big-clique edge (a,b) where a,b in cross_adj[v]
            found = False
            for a in cross_adj[v]:
                for b in cross_adj[v]:
                    if a >= b:
                        continue
                    edge = (a, b)
                    if edge in used_edges:
                        continue
                    # Check if cross edges (a,v) and (b,v) are unused
                    cross_edge_1 = (a, v)
                    cross_edge_2 = (b, v)
                    if cross_edge_1 in used_edges or cross_edge_2 in used_edges:
                        continue
                    # Form triangle
                    cliques.append([a, b, v])
                    used_edges.add(edge)
                    used_edges.add(cross_edge_1)
                    used_edges.add(cross_edge_2)
                    found = True
                    break
                if found:
                    break
        
        # Add remaining edges as 2-cliques
        for e in all_edges:
            if e not in used_edges:
                cliques.append(list(e))
        
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
