def construct():
    import time
    from random import Random
    from collections import defaultdict
    
    deadline = time.time() + 115
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def try_decompose(edges, n, rng, max_restarts=3000):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best = None
        best_count = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random edge ordering
            shuffled = list(edges)
            rng.shuffle(shuffled)
            
            # Current paths: list of lists of vertices
            paths = []
            # Track which edges are used
            used_edges = set()
            # Map from vertex to path index for quick lookup
            # Actually, let's track path endpoints
            
            for u, v in shuffled:
                edge = (min(u,v), max(u,v))
                if edge in used_edges:
                    continue
                used_edges.add(edge)
                
                # Try to extend an existing path at u or v
                merged = False
                for i, path in enumerate(paths):
                    # Can we add u to the end? (path[-1] == u)
                    if path[-1] == u and (u, v) not in used_edges and v not in path[:-1]:
                        # Wait, v might be in path
                        # For simplicity, check if v is not in path
                        if v not in path:
                            paths[i] = path + [v]
                            merged = True
                            break
                    # Can we add v to the end?
                    if path[-1] == v and u not in path:
                        paths[i] = path + [u]
                        merged = True
                        break
                    # Can we add u to the beginning?
                    if path[0] == u and v not in path:
                        paths[i] = [v] + path
                        merged = True
                        break
                    # Can we add v to the beginning?
                    if path[0] == v and u not in path:
                        paths[i] = [u] + path
                        merged = True
                        break
                
                if not merged:
                    paths.append([u, v])
            
            if len(paths) < best_count:
                best_count = len(paths)
                best = [list(p) for p in paths]
            
            if best_count == (len([v for v in range(n) if len(adj[v]) % 2 == 1]) + 1) // 2:
                break
        
        if best is None:
            best = [[list(e)] for e in edges]
        
        return best
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        rng = Random(seed * 7919 + 12345)
        paths = try_decompose(edges, n, rng)
        results.append(paths)
    
    return results
