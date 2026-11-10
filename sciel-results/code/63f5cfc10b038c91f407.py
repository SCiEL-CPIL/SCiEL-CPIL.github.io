def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        N = [set() for _ in range(n)]
        for (u, v) in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)
        
        candidates = []
        
        for v in range(k, n):
            clique = list(N[v]) + [v]
            if len(clique) >= 2:
                candidates.append(frozenset(clique))
                
        I_vertices = list(range(k, n))
        for i in range(len(I_vertices)):
            v1 = I_vertices[i]
            for j in range(i + 1, len(I_vertices)):
                v2 = I_vertices[j]
                common = N[v1].intersection(N[v2])
                clique = list(common) + [v1, v2]
                if len(clique) >= 2:
                    candidates.append(frozenset(clique))
                    
        candidates.append(frozenset(range(k)))
        
        candidates = list(set(candidates))
        
        cand_edges = []
        for c in candidates:
            c_list = list(c)
            es = set()
            for i in range(len(c_list)):
                for j in range(i + 1, len(c_list)):
                    u, v = c_list[i], c_list[j]
                    es.add((min(u, v), max(u, v)))
            cand_edges.append(es)
            
        uncovered = set(edges)
        cliques = []
        
        start_time = time.time()
        
        while uncovered:
            if time.time() - start_time > 110:
                break
                
            best_idx = -1
            best_count = 0
            for i, es in enumerate(cand_edges):
                if len(uncovered) < len(es):
                    count = sum(1 for e in uncovered if e in es)
                else:
                    count = sum(1 for e in es if e in uncovered)
                
                if count > best_count:
                    best_count = count
                    best_idx = i
                    
            if best_idx == -1:
                e = next(iter(uncovered))
                cliques.append(list(e))
                uncovered.discard(e)
                continue
                
            cliques.append(list(candidates[best_idx]))
            uncovered -= cand_edges[best_idx]
            
        # If we timed out, we might have uncovered edges.
        # We must return a valid partition.
        # So we cover remaining edges with individual edges (cliques of size 2)
        while uncovered:
            e = next(iter(uncovered))
            cliques.append(list(e))
            uncovered.discard(e)
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = solve(*spec)
        results.append(cliques)
    
    return results
