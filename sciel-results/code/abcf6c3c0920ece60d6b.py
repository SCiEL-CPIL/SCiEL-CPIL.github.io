def construct():
    from random import Random
    import time
    from collections import deque
    
    start_time = time.time()
    DEADLINE = 130
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose_graph(edges, n):
        if not edges:
            return []
        
        adj = {i: [] for i in range(n)}
        edge_set = set()
        for e in edges:
            u, v = e
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add(tuple(sorted(e)))
        
        odd = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        if not odd:
            return _decompose_even(adj, edge_set, n)
        
        best_paths = None
        best_count = float('inf')
        rng = Random(12345)
        
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE:
                break
            
            perm = odd[:]
            rng.shuffle(perm)
            
            pairs = []
            for i in range(0, len(perm), 2):
                pairs.append((perm[i], perm[i+1]))
            
            paths = []
            remaining = set(edge_set)
            valid = True
            
            for u, v in pairs:
                path = _bfs_path(adj, remaining, u, v)
                if path is None:
                    valid = False
                    break
                paths.append(path)
                for i in range(len(path)-1):
                    e = tuple(sorted((path[i], path[i+1])))
                    if e not in remaining:
                        valid = False
                        break
                    remaining.remove(e)
                if not valid:
                    break
            
            if not valid:
                continue
            
            cycle_paths = _decompose_even(adj, remaining, n)
            if cycle_paths is None:
                continue
            
            all_paths = paths + cycle_paths
            if len(all_paths) < best_count:
                best_count = len(all_paths)
                best_paths = all_paths[:]
        
        if best_paths is None:
            return [list(e) for e in edges]
        return best_paths
    
    def _bfs_path(adj, remaining_edges, start, end):
        if start == end:
            return [start]
        
        parent = {start: None}
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if u == end:
                path = []
                cur = end
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            
            for v in adj[u]:
                e = tuple(sorted((u, v)))
                if v not in parent and e in remaining_edges:
                    parent[v] = u
                    queue.append(v)
        
        return None
    
    def _decompose_even(adj, edge_set, n):
        if not edge_set:
            return []
        
        cycles = []
        remaining = set(edge_set)
        
        while remaining:
            e = next(iter(remaining))
            u, v = e
            remaining.remove(e)
            
            path = [u, v]
            cur = v
            start = u
            
            while cur != start:
                found = False
                for nb in adj[cur]:
                    edge = tuple(sorted((cur, nb)))
                    if edge in remaining:
                        remaining.remove(edge)
                        path.append(nb)
                        cur = nb
                        found = True
                        break
                
                if not found:
                    return None
            
            cycles.append(path)
        
        return cycles
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() - start_time > DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        paths = decompose_graph(edges, n)
        results.append(paths)
    
    return results
