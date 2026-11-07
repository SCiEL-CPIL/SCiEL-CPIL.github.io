"""Strategy: Merge universal neighbors into the core clique to reduce clique count."""

def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _max_clique(vertices, adj_core, available):
        """
        Find a maximum clique in the subgraph induced by 'vertices' 
        where edges are defined by 'available'.
        'adj_core' is not used directly, 'available' is the edge set.
        """
        if not vertices:
            return set()
        
        # If no edges available among vertices, max clique size is 1
        # Check if any edge exists
        has_edge = False
        v_list = list(vertices)
        for i in range(len(v_list)):
            for j in range(i+1, len(v_list)):
                u, v = v_list[i], v_list[j]
                if (min(u, v), max(u, v)) in available:
                    has_edge = True
                    break
            if has_edge:
                break
        
        if not has_edge:
            return {v_list[0]}
        
        # Bron-Kerbosch with pivot
        # R: current clique, P: candidates, X: excluded
        def bk(R, P, X):
            if not P and not X:
                return R
            best = R
            
            # Pivot
            if P:
                # Choose u in P U X maximizing |P intersect N(u)|
                # N(u) in the available graph
                # We need to compute intersection size
                max_deg = -1
                pivot = None
                for u in P.union(X):
                    # Count neighbors in P
                    count = 0
                    for p in P:
                        if (min(u, p), max(u, p)) in available:
                            count += 1
                    if count > max_deg:
                        max_deg = count
                        pivot = u
                
                # Candidates are P \ N(pivot)
                candidates = set()
                for p in P:
                    if (min(pivot, p), max(pivot, p)) not in available:
                        candidates.add(p)
            else:
                candidates = set()
            
            for v in list(candidates):
                new_R = R.union({v})
                new_P = set()
                for p in P:
                    if (min(v, p), max(v, p)) in available:
                        new_P.add(p)
                new_X = set()
                for x in X:
                    if (min(v, x), max(v, x)) in available:
                        new_X.add(x)
                
                res = bk(new_R, new_P, new_X)
                if len(res) > len(best):
                    best = res
                
                P.remove(v)
                X.add(v)
            
            return best
        
        return bk(set(), set(vertices), set())

    def _solve_safe(n, k, p, seed, deadline):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[v].add(u)
        
        # available contains core edges (u, v) with u < v, u, v < k, not yet covered
        available = set((u, v) for u in range(k) for v in range(u + 1, k))
        covered_cross = set()
        cliques = []
        
        # Process each i in I
        # Sort by degree descending? Might help find large cliques early
        # But N_i is fixed.
        for i in range(k, n):
            N_i = set(adj[i])
            while N_i:
                if time.time() > deadline:
                    break
                
                # Find maximum clique S in N_i using available edges
                S = _max_clique(N_i, None, available)
                
                if not S:
                    v = next(iter(N_i))
                    S = {v}
                
                cliques.append([i] + list(S))
                
                # Mark cross edges as covered
                for c in S:
                    covered_cross.add((c, i))
                
                # Remove core edges from available
                for u in S:
                    for v in S:
                        if u < v:
                            available.discard((u, v))
                
                N_i.difference_update(S)
            
            # If broke out of while due to timeout, cover remaining N_i with 2-cliques
            if N_i:
                for c in N_i:
                    if (c, i) not in covered_cross:
                        cliques.append([c, i])
                        covered_cross.add((c, i))
        
        # Now cover remaining core edges
        remaining_edges = set(available)
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: use 2-cliques for all remaining
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break
            
            # Find a maximum clique in the graph defined by remaining_edges
            # Vertices are those incident to remaining edges
            vertices_involved = set()
            for u, v in remaining_edges:
                vertices_involved.add(u)
                vertices_involved.add(v)
            
            S = _max_clique(vertices_involved, None, remaining_edges)
            
            if not S:
                break
            
            if len(S) < 2:
                # Single vertex covers no edges. Pick an edge.
                u, v = next(iter(remaining_edges))
                S = {u, v}
            
            cliques.append(list(S))
            
            # Remove edges in S from remaining_edges
            for u in S:
                for v in S:
                    if u < v:
                        remaining_edges.discard((u, v))
        
        return cliques

    results = []
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(_solve_safe(*spec, deadline))

    return results
