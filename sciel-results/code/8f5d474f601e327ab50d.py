import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        N = {v: set() for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    N[v].add(u)
        return N, k, n

    def get_cliques_for_instance(N, k, n):
        cliques = []
        
        # 1. Cover K-I edges and some K-K edges
        # For each v in I, if N[v] is not empty, use clique N[v] U {v}
        covered_kk_edges = set()
        
        for v in range(k, n):
            neighbors = N[v]
            if len(neighbors) > 0:
                clique = list(neighbors) + [v]
                cliques.append(clique)
                # Mark K-K edges covered by this clique
                # Edges are (u1, u2) where u1, u2 in neighbors
                for u1, u2 in combinations(neighbors, 2):
                    covered_kk_edges.add((min(u1, u2), max(u1, u2)))

        # 2. Cover remaining K-K edges
        # Total possible K-K edges
        all_kk_edges = set()
        for u1 in range(k):
            for u2 in range(u1 + 1, k):
                all_kk_edges.add((u1, u2))
        
        remaining_edges = all_kk_edges - covered_kk_edges
        
        if remaining_edges:
            # Greedy clique cover for remaining edges
            # Build adjacency list for remaining graph
            adj = {u: set() for u in range(k)}
            for u1, u2 in remaining_edges:
                adj[u1].add(u2)
                adj[u2].add(u1)
            
            edges_to_cover = set(remaining_edges)
            
            while edges_to_cover:
                # Find a maximal clique in the graph defined by edges_to_cover
                # Heuristic: Start with vertex of max degree
                active_verts = set()
                for e in edges_to_cover:
                    active_verts.add(e[0])
                    active_verts.add(e[1])
                
                if not active_verts:
                    break
                
                # Pick start vertex with max degree in current subgraph
                start_v = max(active_verts, key=lambda x: len([y for y in adj[x] if (min(x,y), max(x,y)) in edges_to_cover]))
                
                # Greedy expansion
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
                    e = next(iter(edges_to_cover))
                    clique = [e[0], e[1]]
                
                cliques.append(clique)
                
                # Remove covered edges
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
            c = list(set(c))
            if len(c) >= 2:
                c.sort()
                final_cliques.append(c)
        
        results.append(final_cliques)
        
    return results
