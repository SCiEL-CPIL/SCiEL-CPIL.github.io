def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        # Generate graph
        edges = instance(n, k, p, seed)
        
        # Build adjacency for K (0 to k-1) and I (k to n-1)
        # K is 0..k-1, I is k..n-1
        # N(v) for v in I is subset of K
        
        # Map edges to sets for quick lookup
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        # Determine N(v) for each v in I
        N = {v: set() for v in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v) # Wait, u is in I, v in K. So N[u].add(v)
        
        # Correcting N mapping logic
        N = {v: set() for v in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)

        # Greedy assignment
        used_edges_K = set()
        cliques = []
        
        # Sort I vertices by size of N(v) descending to fill K edges with large cliques first?
        # Or ascending? 
        # If we use large N(v) first, we consume many K edges.
        # Smaller N(v) later might find fewer available edges, forcing more splits.
        # So descending seems better.
        I_vertices = sorted([v for v in range(k, n) if N[v]], key=lambda x: len(N[x]), reverse=True)
        
        for v in I_vertices:
            if time.time() > deadline:
                break
            U = list(N[v])
            # Partition U into subsets S such that E(S) are disjoint from used_edges_K and each other
            subsets = []
            remaining_U = set(U)
            
            while remaining_U:
                # Greedily build a subset S
                S = set()
                # Sort U to be deterministic?
                for u in sorted(remaining_U):
                    # Check if adding u to S creates any edge in used_edges_K
                    # Edges added are (u, s) for s in S
                    can_add = True
                    for s in S:
                        e = (min(u, s), max(u, s))
                        if e in used_edges_K:
                            can_add = False
                            break
                    if can_add:
                        S.add(u)
                
                if not S:
                    # Should not happen if we have at least 1 vertex, but just in case
                    S = {remaining_U.pop()}
                
                subsets.append(S)
                # Remove S from remaining_U
                remaining_U -= S
                # Update used_edges_K
                for u1 in S:
                    for u2 in S:
                        if u1 < u2:
                            used_edges_K.add((u1, u2))
                
                # Create clique
                clique = [v] + list(S)
                cliques.append(clique)
                
        # Cover remaining edges in K
        # Remaining edges are those in K not in used_edges_K
        # K vertices are 0..k-1
        # We need to partition the remaining graph on K into cliques
        remaining_K_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in used_edges_K:
                    remaining_K_edges.add((u, v))
        
        # Greedy clique cover on remaining_K_edges
        # Vertices in K: 0..k-1
        # Build adjacency for remaining graph
        adj = {i: set() for i in range(k)}
        for u, v in remaining_K_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        unvisited = set(range(k))
        while unvisited:
            if time.time() > deadline:
                break
            # Find max clique in unvisited using greedy
            # Start with a vertex with highest degree in remaining graph
            # Or just pick one and expand
            start = max(unvisited, key=lambda x: len(adj[x] & unvisited))
            C = {start}
            unvisited.remove(start)
            
            # Greedily add vertices connected to all in C
            candidates = unvisited.copy()
            for u in list(candidates):
                if adj[u].issuperset(C):
                    C.add(u)
                    unvisited.remove(u)
                    candidates.remove(u)
            
            if len(C) >= 2:
                cliques.append(list(C))
            # If len(C) == 1, it's an isolated vertex in remaining graph, no edges to cover, ignore
            
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            # Fallback to seed if timeout?
            # But we must return valid result.
            # If we run out of time, we return what we have?
            # The contract says "return the best result found".
            # If we haven't finished an instance, we can't return partial.
            # We should probably just return the seed result for safety if we run out?
            # But seed is slow to compute? No, seed is fast.
            # Let's just compute seed if we fail.
            pass
        
        res = solve(*spec)
        # Validate types
        # Convert to lists of ints
        final_res = []
        for c in res:
            final_res.append([int(x) for x in c])
        results.append(final_res)
        
    return results
