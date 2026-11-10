    def solve_graph(n, edges):
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        paths = []
        
        if odd_vertices:
            random.shuffle(odd_vertices)
            for i in range(0, len(odd_vertices), 2):
                u = odd_vertices[i]
                v = odd_vertices[i+1]
                
                # BFS
                parent = {u: None}
                q = deque([u])
                while q:
                    curr = q.popleft()
                    if curr == v:
                        break
                    for w in adj[curr]:
                        if w not in parent:
                            parent[w] = curr
                            q.append(w)
                            
                path = []
                curr = v
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                
                paths.append(path)
                
                for i in range(len(path) - 1):
                    a, b = path[i], path[i+1]
                    e = (min(a, b), max(a, b))
                    if e in edge_set:
                        edge_set.remove(e)
                        adj[a].remove(b)
                        adj[b].remove(a)

        # Decompose remaining Eulerian parts
        visited_comp = set()
        for v in range(n):
            if v in visited_comp or not adj[v]:
                continue
            
            # Find component
            comp = set()
            q = deque([v])
            visited_comp.add(v)
            while q:
                curr = q.popleft()
                comp.add(curr)
                for w in adj[curr]:
                    if w not in visited_comp:
                        visited_comp.add(w)
                        q.append(w)
            
            # Find cycles in this component
            while True:
                start = None
                for x in comp:
                    if adj[x]:
                        start = x
                        break
                if start is None:
                    break
                
                # DFS to find cycle
                path = [start]
                visited_path = {start: 0}
                curr = start
                cycle_found = False
                while True:
                    next_found = False
                    for w in adj[curr]:
                        if (min(curr, w), max(curr, w)) not in edge_set:
                            continue
                        if w in visited_path:
                            idx = visited_path[w]
                            cycle = path[idx:]
                            # Process cycle
                            if len(cycle) >= 3:
                                p1 = cycle[:-1]
                                p2 = [cycle[-1], cycle[0]]
                                paths.append(p1)
                                paths.append(p2)
                                
                                for i in range(len(cycle)):
                                    a, b = cycle[i], cycle[(i+1)%len(cycle)]
                                    e = (min(a, b), max(a, b))
                                    if e in edge_set:
                                        edge_set.remove(e)
                                        adj[a].remove(b)
                                        adj[b].remove(a)
                            cycle_found = True
                            break
                        else:
                            path.append(w)
                            visited_path[w] = len(path) - 1
                            curr = w
                            next_found = True
                            break
                    if not next_found:
                        break
                if not cycle_found:
                    break

        return paths
