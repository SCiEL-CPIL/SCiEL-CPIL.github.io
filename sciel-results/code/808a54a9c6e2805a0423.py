def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency for I vertices to C vertices
        # C = 0..k-1, I = k..n-1
        I = list(range(k, n))
        cross_neighbors = {i: [] for i in I}
        
        # We need to know which C-edges are covered by the mixed cliques.
        # A mixed clique for i is {i} U N(i).
        # It covers all edges (u, v) in C such that u in N(i) and v in N(i).
        
        # First, populate cross_neighbors
        for (a, b) in edges:
            if a < k and b < k:
                continue # C-edge, handled later
            elif a >= k and b >= k:
                continue # I-edge, shouldn't exist
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross_neighbors[i].append(c)
        
        mixed_cliques = []
        # Track covered C-edges
        # Using a set of frozensets for C-edges
        covered_C_edges = set()
        
        for i in I:
            neighbors = cross_neighbors[i]
            if not neighbors:
                continue # No cross edges, no mixed clique needed? 
                # But wait, if no cross edges, we don't create a clique.
                # The task says "Every listed set must be a clique... every edge must lie in exactly one listed clique".
                # If i has no edges, it's isolated in the cross-part. It doesn't need to be in a clique?
                # "Every edge must lie in exactly one listed clique".
                # If i has no edges, it's fine if it's not in any clique.
                # However, if i has edges, it MUST be in a clique.
                # So if neighbors is empty, we skip.
                continue
            
            # Create the clique
            clique = [i] + neighbors
            mixed_cliques.append(clique)
            
            # Mark C-edges within neighbors as covered
            for u, v in itertools.combinations(neighbors, 2):
                covered_C_edges.add(frozenset((u, v)))
        
        # Now handle remaining C-edges
        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u + 1, k))
        remaining_C_edges = all_C_edges - covered_C_edges
        
        pure_cliques = []
        if remaining_C_edges:
            # Build adjacency for the remaining graph on C
            # Vertices are 0..k-1
            adj = {u: set() for u in range(k)}
            for e in remaining_C_edges:
                u, v = tuple(e)
                adj[u].add(v)
                adj[v].add(u)
            
            # Greedy Max Clique Cover
            # We want to find cliques in the graph defined by remaining_C_edges.
            # A clique in this graph is a set of vertices where all pairs are in remaining_C_edges.
            
            while remaining_C_edges:
                # Find a maximal clique in the remaining graph
                # Heuristic: Pick a vertex with highest degree, try to extend
                # Or just pick the first vertex and extend greedily
                
                # To find a good clique, we can look for a vertex with high degree
                # and try to form a clique with its neighbors.
                
                best_clique = []
                best_size = 0
                
                # Try to start from each vertex? k is small (10, 20, 30).
                # So O(k^2) or O(k^3) is fine.
                
                for start in range(k):
                    if not adj[start]:
                        continue
                    # Greedy expansion
                    clique = [start]
                    clique_set = {start}
                    candidates = list(adj[start])
                    
                    while candidates:
                        # Pick candidate with max degree in current candidate set?
                        # Or just first one.
                        # Let's pick the one with max degree in the remaining graph restricted to candidates+clique
                        # Simpler: just pick first
                        v = candidates.pop(0)
                        # Check if v is connected to all in clique
                        # Since we built candidates from adj[start], and we filter, 
                        # we need to ensure v is connected to all existing clique members.
                        # Actually, if we just pop, we might pick a v not connected to another clique member.
                        # So we must filter candidates.
                        
                        # Let's restart the expansion logic to be safe
                        pass
                    
                    # Better expansion:
                    clique = [start]
                    clique_set = {start}
                    # Candidates must be connected to all in clique_set
                    candidates = [v for v in adj[start]]
                    
                    while candidates:
                        # Pick v from candidates
                        # To maximize clique size, maybe pick v with max degree in candidates?
                        # Let's just pick the first one for now, or the one with highest degree in adj
                        v = candidates.pop(0)
                        clique.append(v)
                        clique_set.add(v)
                        # Update candidates: must be connected to v and all other clique members
                        # Since v was in candidates, it was connected to all previous clique members.
                        # Now filter candidates to only those connected to v.
                        candidates = [w for w in candidates if w in adj[v]]
                        
                    if len(clique) > best_size:
                        best_size = len(clique)
                        best_clique = clique
                        
                if not best_clique:
                    break # Should not happen if remaining_C_edges is not empty
            
                pure_cliques.append(best_clique)
                
                # Remove edges covered by best_clique from remaining_C_edges and adj
                clique_set = set(best_clique)
                for e in list(remaining_C_edges):
                    if e.issubset(clique_set):
                        remaining_C_edges.discard(e)
                        
                # Update adj
                for u in best_clique:
                    for v in best_clique:
                        if u != v and v in adj[u]:
                            adj[u].discard(v)
                            adj[v].discard(u)
                            
        result = mixed_cliques + pure_cliques
        return result

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
