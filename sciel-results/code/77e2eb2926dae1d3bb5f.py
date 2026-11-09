    def decompose(n, edges):
        from random import Random
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))

        remaining = set(edge_set)
        pieces = []

        def has_edge(a, b):
            e = (a, b) if a < b else (b, a)
            return e in remaining

        def remove_edge(a, b):
            e = (a, b) if a < b else (b, a)
            if e in remaining:
                remaining.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)

        # 1. Find and remove bridges
        visited = {i: False for i in range(n)}
        disc = {i: 0 for i in range(n)}
        low = {i: 0 for i in range(n)}
        bridges = []
        time = 0

        def find_bridges(u, parent):
            nonlocal time
            visited[u] = True
            disc[u] = low[u] = time
            time += 1
            for v in adj[u]:
                if not visited[v]:
                    find_bridges(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.append((u, v))
                elif v != parent:
                    low[u] = min(low[u], disc[v])

        for i in range(n):
            if not visited[i]:
                find_bridges(i, -1)

        for u, v in bridges:
            pieces.append([u, v])
            remove_edge(u, v)

        # 2. Decompose remaining 2-edge-connected components
        def find_long_path(u, v, max_attempts=50):
            best_path = None
            best_len = 0
            rng = Random(42)
            
            for _ in range(max_attempts):
                path = [u]
                onpath = {u}
                current = u
                
                for _step in range(n):
                    if current == v:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_path = path[:]
                        break
                    
                    nbrs = [w for w in adj[current] if has_edge(current, w)]
                    if not nbrs:
                        break
                    
                    ext = [w for w in nbrs if w not in onpath or w == v]
                    if not ext:
                        break
                    
                    if v in ext:
                        p_go_to_v = len(path) / n
                        if rng.random() < p_go_to_v:
                            current = v
                        else:
                            ext_no_v = [w for w in ext if w != v]
                            if not ext_no_v:
                                current = v
                            else:
                                ext_no_v.sort(key=lambda w: len(adj[w]), reverse=True)
                                if rng.random() < 0.7:
                                    current = ext_no_v[0]
                                else:
                                    current = ext_no_v[rng.randint(0, len(ext_no_v)-1)]
                    else:
                        ext.sort(key=lambda w: len(adj[w]), reverse=True)
                        if rng.random() < 0.7:
                            current = ext[0]
                        else:
                            current = ext[rng.randint(0, len(ext)-1)]
                    
                    path.append(current)
                    onpath.add(current)
                    
                if best_len >= n - 1:
                    break

            return best_path

        while remaining:
            edge_to_find = min(remaining)
            u, v = edge_to_find
            
            remove_edge(u, v)
            
            path = find_long_path(u, v)
            
            if path:
                cycle = path + [u]
                pieces.append(cycle)
                
                for i in range(len(path) - 1):
                    a = path[i]
                    b = path[i + 1]
                    remove_edge(a, b)
            else:
                pieces.append([u, v])

        return pieces
