    def try_greedy(k, g, n, seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        # We build vertices 0 to n-1
        # dist[i][j]
        dist = [[-1]*n for _ in range(n)]
        
        # Vertex 0
        # No edges yet
        
        for i in range(1, n):
            # BFS from i to all existing vertices 0..i-1
            # Actually, we need dist between existing vertices too.
            # We can update dist matrix incrementally.
            
            # Initialize dist[i]
            dist[i][i] = 0
            q = deque([i])
            # Note: i is not connected to anyone yet in the graph structure for BFS?
            # No, we are *choosing* neighbors for i.
            # The graph currently has vertices 0..i-1 and their edges.
            # i is isolated.
            # So dist[i][u] is infinity for all u < i initially.
            
            # We need to pick k neighbors u1..uk from 0..i-1.
            # Candidates: u where len(adj[u]) < k.
            candidates = [u for u in range(i) if len(adj[u]) < k]
            
            if len(candidates) < k:
                return None
            
            # We need to pick k candidates such that dist[u][v] >= g-2.
            # We don't know dist[u][v] for u,v < i?
            # We should maintain dist matrix for all pairs.
            
            # Let's maintain dist matrix.
            # When adding i, we don't know its neighbors yet.
            # So we can't compute dist[i][u] until we pick neighbors.
            
            # This creates a chicken-and-egg problem.
            # Standard Greedy:
            # 1. Pick first neighbor u1 randomly from candidates.
            # 2. Pick second neighbor u2 such that dist[u1][u2] >= g-2.
            #    (dist[u1][u2] is known from previous steps).
            # 3. Pick third u3 such that dist[u1][u3] >= g-2 AND dist[u2][u3] >= g-2.
            # ...
            
            # This works if we have dist matrix for existing vertices.
            
            selected = []
            # Shuffle candidates for randomness
            random.shuffle(candidates)
            
            for u in candidates:
                if len(selected) == k:
                    break
                valid = True
                for v in selected:
                    if dist[u][v] < g - 2:
                        valid = False
                        break
                if valid:
                    selected.append(u)
                    adj[i].add(u)
                    adj[u].add(i)
                    # Update dist matrix for new vertex i?
                    # We can do this after all neighbors are picked.
            
            if len(selected) < k:
                return None
            
            # Now update dist for i
            # BFS from i
            dist[i][i] = 0
            q = deque([i])
            while q:
                curr = q.popleft()
                for nb in adj[curr]:
                    if dist[i][nb] == -1:
                        dist[i][nb] = dist[i][curr] + 1
                        dist[nb][i] = dist[i][nb] # Symmetric
                        q.append(nb)
                        
            # Note: dist[u][v] for u,v < i is already correct.
            # dist[i][u] is now computed.
            
        # Check final graph
        # Verify girth
        if check_girth(adj, g):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return edges
        return None
    