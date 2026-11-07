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
        
        # Greedy Max Clique function on a set of vertices with a given available edge set
        def find_max_clique(vertices, avail_edges):
            if len(vertices) <= 1:
                return list(vertices)
            
            # Build local adjacency
            local_adj = {u: set() for u in vertices}
            for u in vertices:
                for w in vertices:
                    if u == w:
                        continue
                    pair = (u, w) if u < w else (w, u)
                    if pair in avail_edges:
                        local_adj[u].add(w)
                        
            best = []
            
            def expand(R, P, X):
                nonlocal best
                if not P and not X:
                    if len(R) > len(best):
                        best = R.copy()
                    return
                
                all_nodes = P.union(X)
                if not all_nodes:
                    return
                
                # Pivot selection
                max_deg = -1
                pivot = None
                for u in all_nodes:
                    deg = len(P.intersection(local_adj.get(u, set())))
                    if deg > max_deg:
                        max_deg = deg
                        pivot = u
                
                # Iterate over P not adjacent to pivot
                if pivot is not None and pivot in P:
                    candidates = P - local_adj.get(pivot, set())
                else:
                    candidates = P.copy()
                    
                for u in list(candidates):
                    expand(R.union({u}), P.intersection(local_adj.get(u, set())), X.intersection(local_adj.get(u, set())))
                    P.remove(u)
                    X.add(u)
                    
            expand(set(), set(vertices), set())
            return best

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
