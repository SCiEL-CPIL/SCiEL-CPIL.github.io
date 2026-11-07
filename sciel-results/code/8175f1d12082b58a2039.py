import random

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = set()
        # Edges within K
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Edges between K and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Separate K and I
        K = list(range(k))
        I = list(range(k, n))
        
        # Determine N(v) for v in I
        N = {v: set() for v in I}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)
        
        # available_edges: set of pairs in K
        available_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                available_edges.add((u, v))
                
        cliques = []
        
        # Sort I by degree descending
        I_sorted = sorted(I, key=lambda v: len(N[v]), reverse=True)
        
        # Helper to remove edges of a clique S from available_edges
        def remove_clique_edges(S):
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    u, v = S_list[i], S_list[j]
                    if u > v:
                        u, v = v, u
                    available_edges.discard((u, v))
        
        # Greedy Max Clique - use Bron-Kerbosch with pivot, optimized
        def find_max_clique(vertices, avail_edges):
            if len(vertices) <= 1:
                return list(vertices)
            
            verts = list(vertices)
            if len(verts) == 2:
                return verts
            
            # Build adjacency as list of sets for speed
            adj = [set() for _ in range(len(verts))]
            idx = {v: i for i, v in enumerate(verts)}
            for u in verts:
                for w in verts:
                    if u == w:
                        continue
                    pair = (u, w) if u < w else (w, u)
                    if pair in avail_edges:
                        adj[idx[u]].add(idx[w])
            
            best = []
            
            def bronk(R, P, X):
                nonlocal best
                if not P and not X:
                    if len(R) > len(best):
                        best = R.copy()
                    return
                
                if not P:
                    return
                
                # Choose pivot u from P ∪ X maximizing |N(u) ∩ P|
                all_nodes = P.union(X)
                if not all_nodes:
                    return
                    
                pivot = max(all_nodes, key=lambda u: len(P.intersection(adj[u])))
                
                # Iterate over P \ N(pivot)
                candidates = P - adj[pivot]
                
                for u in list(candidates):
                    bronk(R.union({u}), P.intersection(adj[u]), X.intersection(adj[u]))
                    P.remove(u)
                    X.add(u)
                    
            bronk(set(), set(range(len(verts))), set())
            return [verts[i] for i in best]

        for v in I_sorted:
            targets = set(N[v])
            while targets:
                # Find largest clique in targets that is a clique in available_edges
                S = find_max_clique(targets, available_edges)
                
                if len(S) >= 2:
                    cliques.append([v] + list(S))
                    remove_clique_edges(S)
                    targets.difference_update(S)
                else:
                    # No clique of size >= 2 found in targets (with available edges)
                    # Take a single vertex
                    u = next(iter(targets))
                    cliques.append([v, u])
                    targets.discard(u)
                    # Note: (v, u) is a cross edge, no K-edge removed.
        
        # Cover remaining edges in K
        while available_edges:
            # Vertices involved in available_edges
            verts_involved = set()
            for u, v in available_edges:
                verts_involved.add(u)
                verts_involved.add(v)
                
            # Find Max Clique in this subgraph
            S = find_max_clique(verts_involved, available_edges)
            
            if len(S) >= 2:
                cliques.append(list(S))
                remove_clique_edges(S)
            else:
                # Pick any edge
                u, v = next(iter(available_edges))
                cliques.append([u, v])
                available_edges.discard((u, v))
                
        # Convert to lists of ints
        # Ensure types are int
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
            
        return final_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_instance(*spec))
        
    return results
