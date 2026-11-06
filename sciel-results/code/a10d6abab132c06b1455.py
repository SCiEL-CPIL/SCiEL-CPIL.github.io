"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    
    def solve_graph(n, edges):
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u,v), max(u,v)))
        
        if not edge_set:
            return []
        
        cliques = []
        remaining = set(edge_set)
        
        # Precompute: for each vertex, the set of neighbors
        # Greedy: repeatedly find a clique that covers many remaining edges
        # Strategy: try triangles first (common in dense graphs), then larger cliques
        
        # Build a list of candidate cliques by expanding from each vertex
        # For each vertex v, find maximal clique in N(v)
        
        def find_clique_in_neighborhood(center, adj, remaining):
            """Find a large clique containing center by expanding in its neighborhood."""
            # Get neighbors of center that have edges in remaining
            candidates = [u for u in adj[center] if (min(center,u), max(center,u)) in remaining]
            if not candidates:
                return None
            
            # Greedy expand: pick a neighbor, then find common neighbors, etc.
            # Try to build a clique by iterative selection
            clique = [center]
            available = set(candidates)
            
            while available:
                # Pick vertex in available with most connections within available
                best = None
                best_count = -1
                for u in available:
                    count = len(available & adj[u])
                    if count > best_count:
                        best_count = count
                        best = u
                if best is None:
                    break
                clique.append(best)
                # Remove vertices not adjacent to best
                available = available & adj[best]
                available.discard(best)
            
            if len(clique) < 2:
                return None
            return clique
        
        # Iteratively find cliques covering remaining edges
        # To be efficient, limit iterations
        max_iters = 5000
        iter_count = 0
        
        while remaining and iter_count < max_iters:
            iter_count += 1
            
            # Find the clique that covers the most remaining edges
            # Sample some vertices to find good cliques
            best_clique = None
            best_cover = 0
            
            # Try from each vertex (or sample if too many)
            vertices_to_try = list(range(n))
            if n > 40:
                # Sample a subset for speed
                import random
                rng_sample = Random(iter_count * 7919)
                vertices_to_try = rng_sample.sample(vertices_to_try, min(n, 30))
            
            for center in vertices_to_try:
                clique = find_clique_in_neighborhood(center, adj, remaining)
                if clique is None:
                    continue
                # Count how many remaining edges this clique covers
                cover = 0
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        if e in remaining:
                            cover += 1
                if cover > best_cover:
                    best_cover = cover
                    best_clique = clique
            
            if best_clique is None or best_cover == 0:
                # No clique found via this method; fall back to single edges
                # Find any remaining edge
                e = next(iter(remaining))
                best_clique = [e[0], e[1]]
                best_cover = 1
            
            cliques.append(best_clique)
            # Remove all edges in this clique from remaining
            for i in range(len(best_clique)):
                for j in range(i+1, len(best_clique)):
                    e = (min(best_clique[i], best_clique[j]), max(best_clique[i], best_clique[j]))
                    remaining.discard(e)
        
        # Handle any leftover edges as single-edge cliques
        for e in remaining:
            cliques.append([e[0], e[1]])
        
        return cliques
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = solve_graph(n, edges)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
