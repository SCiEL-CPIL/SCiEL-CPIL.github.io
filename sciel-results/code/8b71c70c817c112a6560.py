def construct():
    import time
    deadline = time.time() + 140  # safety margin
    
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def solve(edges, n):
        edge_set = set()
        for (u,v) in edges:
            edge_set.add(frozenset((u,v)))
        adj = [set() for _ in range(n)]
        for (u,v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        uncovered = set(edge_set)
        cliques = []
        
        while uncovered:
            # Find a clique in original graph with all edges uncovered, size >= 2
            # Greedy: pick a vertex with high degree in uncovered subgraph
            # Compute degree in uncovered subgraph
            deg = [0]*n
            for e in uncovered:
                a,b = tuple(e)
                deg[a]+=1
                deg[b]+=1
            # Start from highest degree vertex
            start = max(range(n), key=lambda v: deg[v])
            # Build clique greedily
            clique = [start]
            candidates = [u for u in adj[start] if frozenset((start,u)) in uncovered]
            while candidates:
                # Pick candidate with max common neighbors in candidates
                best = max(candidates, key=lambda u: sum(1 for w in candidates if w in adj[u]))
                # Check all edges from best to current clique are uncovered
                ok = True
                for c in clique:
                    if frozenset((best,c)) not in uncovered:
                        ok = False
                        break
                if not ok:
                    candidates.remove(best)
                    continue
                clique.append(best)
                # Remove from candidates those not connected to best via uncovered edge
                candidates = [u for u in candidates if u != best and frozenset((best,u)) in uncovered and u in adj[best]]
            if len(clique) >= 2:
                # Assign all edges of this clique
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        uncovered.discard(frozenset((clique[i], clique[j])))
                cliques.append(clique)
            else:
                # Should not happen if uncovered non-empty, but fallback
                # Take any uncovered edge
                e = next(iter(uncovered))
                a,b = tuple(e)
                cliques.append([a,b])
                uncovered.discard(e)
        return cliques
    
    results = []
    for (n, p, seed) in ((32,0.6,601),(64,0.55,602)):
        if time.time() > deadline:
            # fallback
            edges = instance(n,p,seed)
            results.append([list(e) for e in edges])
            continue
        edges = instance(n,p,seed)
        cliques = solve(edges, n)
        results.append(cliques)
    return results
