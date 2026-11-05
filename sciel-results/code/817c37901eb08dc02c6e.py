    def try_random_repair(n, k, g, timeout=10):
        start_time = time.time()
        
        def generate_random_regular():
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    return None
                adj[u].append(v)
                adj[v].append(u)
            # Check for multi-edges
            for u in range(n):
                if len(adj[u]) != len(set(adj[u])):
                    return None
            return adj

        def find_cycle_and_edges(adj, g):
            # Returns (cycle_nodes, cycle_edges) or None
            # BFS from 0
            for start in range(n):
                visited = {start: 0}
                parent = {start: -1}
                queue = deque([(start, 0)])
                while queue:
                    curr, depth = queue.popleft()
                    if depth >= (g-1)//2:
                        continue
                    for nei in adj[curr]:
                        if nei == parent[curr]:
                            continue
                        if nei in visited:
                            # Cycle found
                            # Reconstruct cycle
                            # This is complex.
                            return True # Just return True if any short cycle exists
                        else:
                            visited[nei] = depth + 1
                            parent[nei] = curr
                            queue.append((nei, depth + 1))
            return False

        # Simple Repair:
        # 1. Generate
        # 2. If valid, return
        # 3. If not, try 1000 random 2-switches and check again.
        
        adj = generate_random_regular()
        if not adj:
            return None
            
        # Check if valid
        def is_valid(adj):
            # Check girth
            for u in range(n):
                visited = {u: 0}
                queue = deque([(u, 0)])
                while queue:
                    curr, depth = queue.popleft()
                    if depth >= (g-1)//2:
                        continue
                    for nei in adj[curr]:
                        if nei in visited:
                            if depth + visited[nei] + 1 < g:
                                return False
                        else:
                            visited[nei] = depth + 1
                            queue.append((nei, depth + 1))
            return True

        if is_valid(adj):
            edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
            return n, edges

        # Repair loop
        for _ in range(100):
            if time.time() - start_time > timeout:
                break
            # Pick random 2-switch
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if v not in adj[u]: continue
            # Find w, x such that (w,x) is an edge, w!=u, x!=v
            # Swap (u,v) and (w,x) with (u,w) and (v,x)
            # This is heuristic.
            
            # Just restart if too hard.
            adj = generate_random_regular()
            if not adj:
                continue
            if is_valid(adj):
                edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
                return n, edges

        return None
