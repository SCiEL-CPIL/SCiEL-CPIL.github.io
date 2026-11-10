def construct():
    import time
    from random import Random

    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed):
        cross_edges = instance(n, k, p, seed)

        # N[v] is list of neighbors in C for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)

        # All edges in C
        all_c_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_c_edges.add(frozenset((u, w)))

        covered = set()
        cliques = []

        # Step 1: Create maximal mixed cliques
        # For each v in I, the clique is {v} U N(v)
        # This covers all cross edges incident to v
        # And all edges between vertices in N(v)
        for v in range(k, n):
            neighbors = N[v]
            if not neighbors:
                continue
            
            # Add the clique
            clique = [v] + neighbors
            cliques.append(clique)
            
            # Mark edges within neighbors as covered
            # Note: We only care about edges in C
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    u = neighbors[i]
                    w = neighbors[j]
                    covered.add(frozenset((u, w)))

        # Step 2: Cover remaining C-edges
        rem_edges = set()
        for e in all_c_edges:
            if e not in covered:
                rem_edges.add(e)
        
        # Greedy clique cover for rem_edges
        # To speed up, we can use a simple greedy:
        # Pick a vertex with max degree, form maximal clique, remove edges.
        # Since k is small, we can afford a bit more logic.
        
        # Adjacency list for rem_edges
        adj = {i: set() for i in range(k)}
        for e in rem_edges:
            u, w = tuple(e)
            adj[u].add(w)
            adj[w].add(u)
            
        while rem_edges:
            if time.time() > deadline - 1.0:
                # Fallback: add all remaining edges as size-2 cliques
                for e in rem_edges:
                    u, w = tuple(e)
                    cliques.append([u, w])
                break

            # Find a maximal clique
            # Heuristic: Pick vertex with highest degree in remaining graph
            # If degrees are 0, pick any vertex in rem_edges
            max_deg = -1
            start_v = None
            for v in range(k):
                if len(adj[v]) > max_deg:
                    max_deg = len(adj[v])
                    start_v = v
            
            if start_v is None or max_deg == 0:
                 # Should not happen if rem_edges is not empty, 
                 # but if adj is empty, we need to find an edge
                 # Actually if rem_edges is not empty, some adj must be non-empty
                 # Unless we have a bug.
                 # Let's just pick an arbitrary edge
                 e = next(iter(rem_edges))
                 u, w = tuple(e)
                 clique = [u, w]
            else:
                # Greedily build clique starting from start_v
                S = {start_v}
                # Candidates must be neighbors of all in S
                cand = set(adj[start_v])
                
                while cand:
                    # Pick neighbor with max degree in cand?
                    # Or just any?
                    # Let's pick one that is connected to all current S (which is guaranteed by cand definition)
                    # To make it "maximal", we just need to ensure we don't stop early.
                    # Standard greedy: add vertex, update cand to intersection of neighbors.
                    
                    # Pick v with max degree in cand to grow clique faster?
                    best_v = None
                    best_deg = -1
                    for v in cand:
                        if len(adj[v]) > best_deg:
                            best_deg = len(adj[v])
                            best_v = v
                            
                    if best_v is None:
                        break
                        
                    S.add(best_v)
                    # New cand is intersection of neighbors of all in S
                    new_cand = set(adj[best_v])
                    for v in S:
                        if v != best_v:
                            new_cand &= adj[v]
                    cand = new_cand

                clique = list(S)
            
            if len(clique) < 2:
                 # Safety: if we couldn't form a clique, take an edge
                 e = next(iter(rem_edges))
                 u, w = tuple(e)
                 clique = [u, w]
                 
            cliques.append(clique)
            
            # Remove all edges within clique from rem_edges and adj
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u = clique[i]
                    w = clique[j]
                    e = frozenset((u, w))
                    if e in rem_edges:
                        rem_edges.discard(e)
                        adj[u].discard(w)
                        adj[w].discard(u)
                        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []

    for (n, k, p, seed) in specs:
        if time.time() > deadline - 5.0:
             # Emergency fallback
             ce = instance(n, k, p, seed)
             cliques = [[int(u) for u in range(k)]]
             for (u, v) in ce:
                 cliques.append([int(u), int(v)])
             result.append(cliques)
             continue
             
        cliques = solve_instance(n, k, p, seed)
        # Ensure types
        clean = []
        for clique in cliques:
            clean.append([int(x) for x in clique])
        result.append(clean)

    return result
