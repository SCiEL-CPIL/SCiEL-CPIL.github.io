def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def find_maximal_clique(vertices, edge_set):
        """
        Find a maximal clique in the subgraph induced by 'vertices' using edges in 'edge_set'.
        vertices: list of ints
        edge_set: set of tuples (min, max)
        Returns list of vertices in the clique.
        """
        if not vertices:
            return []
        
        # Calculate degrees within the induced subgraph
        degs = {v: 0 for v in vertices}
        v_set = set(vertices)
        for u, v in edge_set:
            if u in v_set and v in v_set:
                degs[u] += 1
                degs[v] += 1
        
        # Sort by degree descending
        sorted_v = sorted(vertices, key=lambda x: degs[x], reverse=True)
        
        clique = []
        for v in sorted_v:
            # Check if v is connected to all in current clique
            is_connected = True
            for c in clique:
                edge = (min(v, c), max(v, c))
                if edge not in edge_set:
                    is_connected = False
                    break
            if is_connected:
                clique.append(v)
                
        return clique

    def greedy_clique_partition(edge_set, n):
        """
        Partition edges in edge_set into cliques.
        """
        if not edge_set:
            return []
            
        cliques = []
        remaining = set(edge_set)
        
        # Build adjacency for quick lookup? 
        # We are using edge_set directly.
        
        while remaining:
            # Pick a vertex with max degree in remaining
            deg = {}
            for u, v in remaining:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            start = max(deg, key=lambda x: deg[x])
            
            # Find maximal clique containing start in remaining
            # Candidates are neighbors of start in remaining
            candidates = set()
            for u, v in remaining:
                if u == start:
                    candidates.add(v)
                elif v == start:
                    candidates.add(u)
            
            # We need a clique containing start.
            # Let's use the find_maximal_clique logic but force start to be in it.
            # Simpler: use find_maximal_clique on [start] + list(candidates)
            # But find_maximal_clique might not pick start if another vertex has higher degree.
            # We want a maximal clique *containing* start to ensure we remove edges incident to start.
            
            # Greedy build clique containing start
            clique = [start]
            # Sort candidates by degree in remaining
            cands_list = list(candidates)
            cands_list.sort(key=lambda x: deg.get(x, 0), reverse=True)
            
            for c in cands_list:
                connected = True
                for m in clique:
                    if (min(c, m), max(c, m)) not in remaining:
                        connected = False
                        break
                if connected:
                    clique.append(c)
            
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                # Remove edges
                to_remove = set()
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        u, v = clique[i], clique[j]
                        to_remove.add((min(u, v), max(u, v)))
                remaining -= to_remove
            else:
                # Fallback: take one edge
                u, v = next(iter(remaining))
                cliques.append([u, v])
                remaining.discard((min(u, v), max(u, v)))
                
        return cliques

    def split_graph_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        # C is 0..k-1, I is k..n-1
        C = list(range(k))
        I = list(range(k, n))
        
        # Precompute neighbors in I for each vertex? 
        # We need to process I vertices.
        
        # 1. Process I vertices
        # We need to be careful about the order. 
        # Processing vertices with higher degree first might cover more C-C edges?
        # Or does it matter? 
        # Let's sort I by degree in the original graph descending.
        
        degs = {v: 0 for v in range(n)}
        for u, v in edge_set:
            degs[u] += 1
            degs[v] += 1
            
        I_sorted = sorted(I, key=lambda x: degs[x], reverse=True)
        
        cliques = []
        remaining = set(edge_set)
        
        # Helper to get neighbors in remaining
        def get_neighbors_in_remaining(v):
            nbrs = set()
            for u, w in remaining:
                if u == v:
                    nbrs.add(w)
                elif w == v:
                    nbrs.add(u)
            return nbrs

        for v in I_sorted:
            nbrs = get_neighbors_in_remaining(v)
            if not nbrs:
                continue
            
            # We want to form a clique {v} U S where S is a clique in remaining[C]
            # S must be subset of nbrs (which are in C)
            # Find maximal clique in remaining induced by nbrs
            # Note: nbrs are all in C.
            
            # Use find_maximal_clique on nbrs with edge_set=remaining
            # But we need to ensure v is included? 
            # The clique will be {v} U S.
            # S is a clique in remaining.
            # Edges in {v} U S are:
            # 1. (v, u) for u in S. These are in remaining because S subset nbrs.
            # 2. (u, w) for u, w in S. These are in remaining because S is a clique in remaining.
            
            S = find_maximal_clique(list(nbrs), remaining)
            
            if S:
                clique = [v] + S
                cliques.append(sorted(clique))
                
                # Remove edges from remaining
                to_remove = set()
                # Edges incident to v
                for u in S:
                    to_remove.add((min(v, u), max(v, u)))
                # Edges within S
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        u, w = S[i], S[j]
                        to_remove.add((min(u, w), max(u, w)))
                        
                remaining -= to_remove
            else:
                # If S is empty, it means no neighbor is connected to any other?
                # Or nbrs was empty?
                # If nbrs is not empty, S should have at least 1 vertex (the one with max degree).
                # Wait, find_maximal_clique returns [] if vertices is empty.
                # If nbrs is not empty, S will have at least 1 vertex.
                # So this case shouldn't happen if nbrs is not empty.
                pass

        # 2. Process remaining edges (all in C)
        if remaining:
            # Use greedy_clique_partition on remaining
            # We need to map vertices back? No, they are already 0..k-1
            # But greedy_clique_partition expects n? It uses range(n) for deg calc?
            # My greedy_clique_partition above uses deg dict, so it's fine.
            # But it assumes vertices are in 0..n-1.
            # The remaining edges are only in C (0..k-1).
            # So we can just call it.
            
            # However, greedy_clique_partition might pick a vertex from I if it exists in remaining?
            # No, all edges incident to I are removed.
            # So remaining only has edges in C.
            
            cliques_C = greedy_clique_partition(remaining, k) # n=k is enough? 
            # The function uses range(n) in max()? 
            # In my code: start = max(deg, key=lambda x: deg[x]). deg keys are only those in remaining.
            # So it's safe.
            cliques.extend(cliques_C)
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = split_graph_partition(n, k, p, seed)
        results.append(cliques)
        
    return results
