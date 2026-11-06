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

    def longest_simple_path(adj, n):
        best_path = []
        
        def dfs(v, visited, path):
            nonlocal best_path
            if len(path) > len(best_path):
                best_path = path[:]
            # Order neighbors by degree descending
            neighbors = sorted(adj.get(v, []), key=lambda x: len(adj.get(x, [])), reverse=True)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)
        
        # Start from high-degree vertices
        starts = sorted(range(n), key=lambda x: len(adj.get(x, [])), reverse=True)
        for start in starts:
            if adj.get(start):
                dfs(start, {start}, [start])
        
        return best_path

    def decompose_into_paths(n, edges):
        if not edges:
            return []
        
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(set)
        for u, v in remaining_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        while remaining_edges:
            path = longest_simple_path(adj, n)
            if not path or len(path) < 2:
                break
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = tuple(sorted((u, v)))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            paths.append(path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        n = spec[0]
        paths = decompose_into_paths(n, edges)
        results.append(paths)

    return results

