def construct():
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_cycle(adj, edges_set):
        visited = {}
        parent = {}
        
        for start in sorted(adj.keys()):
            if visited.get(start, 0) != 0:
                continue
            stack = [(start, iter(sorted(adj[start])))]
            visited[start] = 1
            parent[start] = None
            while stack:
                node, neighbors = stack[-1]
                advanced = False
                for neighbor in neighbors:
                    e = tuple(sorted((node, neighbor)))
                    if e not in edges_set:
                        continue
                    if visited.get(neighbor, 0) == 0:
                        visited[neighbor] = 1
                        parent[neighbor] = node
                        stack.append((neighbor, iter(sorted(adj[neighbor]))))
                        advanced = True
                        break
                    elif visited.get(neighbor, 0) == 1:
                        # Back edge found
                        cycle = [node]
                        cur = node
                        while cur != neighbor:
                            cur = parent[cur]
                            if cur is None:
                                break
                            cycle.append(cur)
                        if cur != neighbor:
                            continue
                        # cycle = [node, ..., neighbor]
                        if len(cycle) < 3:
                            continue
                        return cycle
                if not advanced:
                    stack.pop()
                    visited[node] = 2
        
        return None
    
    def decompose(edges):
        if not edges:
            return []
        max_v = max(max(e) for e in edges)
        adj = {v: set() for v in range(max_v + 1)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        edges_set = {tuple(sorted(e)) for e in edges}
        pieces = []
        
        while edges_set:
            cycle = find_cycle(adj, edges_set)
            if cycle is None:
                break
            # Verify
            if len(set(cycle)) < 3:
                break
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                if tuple(sorted((u, v))) not in edges_set:
                    valid = False
                    break
            if not valid:
                break
            pieces.append(list(cycle))
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                edges_set.discard(tuple(sorted((u, v))))
        
        for e in edges_set:
            pieces.append(list(e))
        
        return pieces
    
    return [decompose(instance(*spec)) for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))]
