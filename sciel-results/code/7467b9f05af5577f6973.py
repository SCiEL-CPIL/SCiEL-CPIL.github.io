import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        # K is 0..k-1, I is k..n-1
        # Edges in K
        # Edges between K and I
        N = {v: set() for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    N[v].add(u)
        return N, k, n

    def get_cliques_for_instance(N, k, n):
        cliques = []
        
        # 1. Select S subset of I
        # Candidates: v in I with |N[v]| >= 2
        candidates = [v for v in range(k, n) if len(N[v]) >= 2]
        # Sort by size descending
        candidates.sort(key=lambda v: len(N[v]), reverse=True)
        
        S = []
        # To check compatibility, we need to check intersection of N[u] and N[v]
        # Constraint: |N[u] & N[v]| <= 1
        
        for v in candidates:
            compatible = True
            for u in S:
                if len(N[u].intersection(N[v])) > 1:
                    compatible = False
                    break
            if compatible:
                S.append(v)
        
        # 2. Create cliques for v in S
        # C_v = {v} U N[v]
        edges_covered_in_K = set()
        for v in S:
            clique = [v] + list(N[v])
            # Ensure size >= 2
            if len(clique) >= 2:
                cliques.append(clique)
            # Update edges_covered_in_K
            for u1, u2 in combinations(N[v], 2):
                edges_covered_in_K.add((min(u1, u2), max(u1, u2)))
        
        # 3. Create cliques for v not in S
        # For each v not in S, cover each edge (v, u) with {v, u}
        S_set = set(S)
        for v in range(k, n):
            if v not in S_set:
                for u in N[v]:
                    cliques.append([v, u])
        
        # 4. Cover remaining edges in K
        # All edges in K are (u1, u2) with u1 < u2 < k
        # Remove those in edges_covered_in_K
        remaining_edges = []
        for u1 in range(k):
            for u2 in range(u1 + 1, k):
                if (u1, u2) not in edges_covered_in_K:
                    remaining_edges.append((u1, u2))
        
        # Greedy clique cover for remaining_edges
        # Build adjacency for remaining graph
        # Vertices are 0..k-1
        # Edge exists if in remaining_edges
        adj = {u: set() for u in range(k)}
        for u1, u2 in remaining_edges:
            adj[u1].add(u2)
            adj[u2].add(u1)
        
        edges_to_cover = set(remaining_edges)
        
        while edges_to_cover:
            # Find largest clique in the graph defined by edges_to_cover
            # Simple heuristic: try to extend from each vertex
            best_clique = []
            # Convert edges to list for iteration
            # Since k is small, we can check subsets
            # Or just pick a vertex with max degree in remaining graph and try to build clique
            # Let's just find a maximal clique greedily
            # Pick a vertex
            verts = list(adj.keys())
            # Actually, we should pick from vertices involved in edges_to_cover
            active_verts = set()
            for e in edges_to_cover:
                active_verts.add(e[0])
                active_verts.add(e[1])
            
            if not active_verts:
                break
                
            # Try to find a large clique starting from each active vertex
            # To save time, just pick one with high degree
            start_v = max(active_verts, key=lambda x: len([y for y in adj[x] if (min(x,y), max(x,y)) in edges_to_cover]))
            
            # Greedy clique construction
            clique = [start_v]
            candidates = sorted(adj[start_v], key=lambda x: len([y for y in adj[x] if (min(x,y), max(x,y)) in edges_to_cover]), reverse=True)
            for c in candidates:
                # Check if c is connected to all in clique
                ok = True
                for u in clique:
                    if (min(u, c), max(u, c)) not in edges_to_cover:
                        ok = False
                        break
                if ok:
                    clique.append(c)
            
            if len(clique) < 2:
                # Should not happen if edges_to_cover not empty
                # Fallback: take first edge
                e = next(iter(edges_to_cover))
                clique = [e[0], e[1]]
            
            cliques.append(clique)
            
            # Remove edges covered by this clique
            for u1, u2 in combinations(clique, 2):
                e = (min(u1, u2), max(u1, u2))
                edges_to_cover.discard(e)
                
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        n, k, p, seed = spec
        N, k_val, n_val = instance(n, k, p, seed)
        cliques = get_cliques_for_instance(N, k_val, n_val)
        
        # Validation: Convert to lists of ints
        final_cliques = []
        for c in cliques:
            # Ensure distinct ints
            c = list(set(c))
            if len(c) >= 2:
                # Sort for consistency? Not required but good
                c.sort()
                final_cliques.append(c)
        
        results.append(final_cliques)
        
    return results
