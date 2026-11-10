# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        covered_internal = set()
        
        # For each independent vertex, create clique {i} + N(i)
        for i in range(k, n):
            Ni = [u for u in range(k) if u in adj[i]]
            if len(Ni) >= 1:
                clique = [i] + Ni
                cliques.append(clique)
                for a in range(len(Ni)):
                    for b in range(a + 1, len(Ni)):
                        covered_internal.add((Ni[a], Ni[b]))
        
        # Find uncovered internal edges
        remaining = set()
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in covered_internal:
                    remaining.add((u, v))
        
        # Cover remaining internal edges with cliques of clique vertices
        # Greedy: repeatedly find a maximal clique in the remaining edge graph
        H_adj = [set() for _ in range(k)]
        for u, v in remaining:
            H_adj[u].add(v)
            H_adj[v].add(u)
        
        while remaining:
            # Find vertex with max degree in remaining
            degs = {}
            for u, v in remaining:
                degs[u] = degs.get(u, 0) + 1
                degs[v] = degs.get(v, 0) + 1
            if not degs:
                break
            start = max(degs, key=degs.get)
            
            # Build maximal clique greedily
            clique = [start]
            candidates = set(H_adj[start]) & set(degs.keys())
            changed = True
            while changed:
                changed = False
                for c in list(candidates):
                    if all(c in H_adj[x] for x in clique):
                        clique.append(c)
                        candidates.remove(c)
                        changed = True
                        break
            
            if len(clique) >= 2:
                cliques.append(clique)
                clique_set = set(clique)
                remaining = {(u, v) for u, v in remaining if not (u in clique_set and v in clique_set)}
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        results.append(cliques)
        
        if time.time() > deadline:
            break
    
    return results
# EVOLVE-BLOCK-END
