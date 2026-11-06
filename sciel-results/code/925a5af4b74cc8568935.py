def construct():
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 145

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))
            
        cliques = []
        remaining_edges = set(edge_set)
        
        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques.append(list(e))
                break
                
            deg = {v: 0 for v in range(n)}
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            v_start = max(deg, key=lambda x: deg[x])
            
            clique = [v_start]
            
            candidates = set()
            for u, v in remaining_edges:
                if u == v_start:
                    candidates.add(v)
                elif v == v_start:
                    candidates.add(u)
            
            while candidates:
                u_next = max(candidates, key=lambda x: deg[x])
                clique.append(u_next)
                
                new_candidates = set()
                for u, v in remaining_edges:
                    if u == u_next:
                        if v in candidates:
                            new_candidates.add(v)
                    elif v == u_next:
                        if u in candidates:
                            new_candidates.add(u)
                
                candidates = new_candidates
                
            for u, v in combinations(clique, 2):
                e = (u, v) if u < v else (v, u)
                if e in remaining_edges:
                    remaining_edges.remove(e)
                    
            cliques.append(clique)
            
        return cliques

    c1 = solve(random_graph(32, 0.6, 601))
    c2 = solve(random_graph(64, 0.55, 602))
    
    return [c1, c2]
