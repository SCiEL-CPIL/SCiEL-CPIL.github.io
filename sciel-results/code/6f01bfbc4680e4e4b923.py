def construct():
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        while remaining:
            # Find a vertex with max degree in remaining graph
            deg = {}
            for u, v in remaining:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            # Start from highest degree vertex
            start = max(deg, key=lambda x: deg[x])
            
            # Greedy extension
            path = [start]
            used_vertices = {start}
            # At each end, we can extend
            # We'll maintain the path and try to extend at either end
            
            extended = True
            while extended:
                extended = False
                # Try extending at the back
                last = path[-1]
                neighbors = [n for n in adj.get(last, []) if n not in used_vertices]
                # Filter to only edges that are in remaining
                valid_neighbors = []
                for n in neighbors:
                    edge = tuple(sorted((last, n)))
                    if edge in remaining:
                        valid_neighbors.append(n)
                
                if valid_neighbors:
                    # Pick the neighbor with highest degree
                    best = max(valid_neighbors, key=lambda x: deg.get(x, 0))
                    path.append(best)
                    used_vertices.add(best)
                    remaining.remove(tuple(sorted((last, best))))
                    extended = True
                else:
                    # Try extending at the front
                    first = path[0]
                    neighbors = [n for n in adj.get(first, []) if n not in used_vertices]
                    valid_neighbors = []
                    for n in neighbors:
                        edge = tuple(sorted((first, n)))
                        if edge in remaining:
                            valid_neighbors.append(n)
                    
                    if valid_neighbors:
                        best = max(valid_neighbors, key=lambda x: deg.get(x, 0))
                        path.insert(0, best)
                        used_vertices.add(best)
                        remaining.remove(tuple(sorted((first, best))))
                        extended = True
            
            # Record the path
            # Convert to list of vertices
            paths.append(path)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results
