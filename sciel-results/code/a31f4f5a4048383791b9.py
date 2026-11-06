def construct():
    from random import Random
    from collections import defaultdict
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges, rng):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining = set()
        for u, v in edges:
            remaining.add((u, v))
            remaining.add((v, u))
        
        paths = []
        
        def greedy_path(start):
            path = [start]
            visited = {start}
            while True:
                candidates = []
                for nb in adj[path[-1]]:
                    if (path[-1], nb) in remaining and nb not in visited:
                        deg = sum(1 for x in adj[nb] if (nb, x) in remaining)
                        candidates.append((deg, nb))
                if not candidates:
                    break
                # Pick best, with some randomness for diversity
                candidates.sort(reverse=True)
                # Sometimes pick second best for diversity
                if len(candidates) > 1 and rng.random() < 0.3:
                    best_nb = candidates[1][1]
                else:
                    best_nb = candidates[0][1]
                remaining.discard((path[-1], best_nb))
                remaining.discard((best_nb, path[-1]))
                path.append(best_nb)
                visited.add(best_nb)
            return path
        
        while remaining:
            deg = {v: sum(1 for x in adj[v] if (v, x) in remaining) for v in adj}
            # Pick start with highest degree, random tie-break
            max_deg = max(deg.values())
            starts = [v for v in deg if deg[v] == max_deg]
            start = rng.choice(starts)
            path = greedy_path(start)
            paths.append(path)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        best_paths = None
        best_count = float('inf')
        for trial in range(50):
            rng = Random(trial)
            paths = decompose(edges, rng)
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
        results.append(best_paths)
    
    return results
