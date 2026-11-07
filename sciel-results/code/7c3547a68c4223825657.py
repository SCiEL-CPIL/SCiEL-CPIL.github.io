    def get_embedding_generator(tree_idx, free_edges):
        k = tree_idx + 1
        if k == 1:
            yield [0]
            return
        if k == 2:
            for e in free_edges:
                u, v = tuple(e)
                yield [u, v]
            return
            
        adj = tree_adjs[tree_idx]
        parent = [-1] * k
        children = [[] for _ in range(k)]
        visited = [False] * k
        visited[0] = True
        stack = [0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    children[u].append(v)
                    stack.append(v)
                    
        depth = [0] * k
        for i in range(1, k):
            depth[i] = depth[parent[i]] + 1
            
        order = sorted(range(k), key=lambda x: (depth[x], -len(adj[x]), x))
        
        host = [-1] * k
        used = set()
        
        # Pre-compute free degrees
        free_deg = [0] * n
        for e in free_edges:
            u, v = tuple(e)
            free_deg[u] += 1
            free_deg[v] += 1

        def dfs(idx):
            if idx == len(order):
                yield list(host)
                return
                
            v = order[idx]
            p = parent[v]
            
            if p == -1:
                cands = list(range(n))
            else:
                hp = host[p]
                cands = [h for h in range(n) if h != hp and frozenset((hp, h)) in free_edges]
                
            # Sort by free_deg descending
            cands.sort(key=lambda h: free_deg[h], reverse=True)
            
            for h in cands:
                if h in used:
                    continue
                host[v] = h
                used.add(h)
                
                # Forward checking
                if children[v]:
                    has_free = False
                    for e in free_edges:
                        if h in e:
                            u2, v2 = tuple(e)
                            other = v2 if u2 == h else u2
                            if other not in used:
                                has_free = True
                                break
                    if not has_free:
                        host[v] = -1
                        used.remove(h)
                        continue
                        
                for sol in dfs(idx + 1):
                    yield sol
                
                host[v] = -1
                used.remove(h)

        yield from dfs(0)
