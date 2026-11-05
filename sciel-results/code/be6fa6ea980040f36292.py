def construct():
    import time
    
    deadline = time.time() + 140
    
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def build_adj(n, edges):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def bfs_dist(adj, src, n):
        """BFS from src, return distance dict."""
        from collections import deque
        dist = {src: 0}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist
    
    def bfs_path(adj, src, dst, n):
        """Find shortest path from src to dst."""
        from collections import deque
        if src == dst:
            return [src]
        visited = {src}
        parent = {}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in sorted(adj[u]):
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    if v == dst:
                        path = [v]
                        while path[-1] != src:
                            path.append(parent[path[-1]])
                        path.reverse()
                        return path
                    queue.append(v)
        return None
    
    def eulerian_decompose(adj, n):
        """Decompose an Eulerian graph (all even degrees) into cycles.
        Returns list of cycles (each cycle is a list of vertices, first != last)."""
        cycles = []
        adj_copy = {v: set(s) for v, s in adj.items()}
        
        while any(adj_copy[v] for v in adj_copy):
            # Find a vertex with remaining edges
            start = None
            for v in range(n):
                if adj_copy[v]:
                    start = v
                    break
            if start is None:
                break
            
            # Hierholzer's algorithm
            stack = [start]
            cycle = []
            while stack:
                v = stack[-1]
                if adj_copy[v]:
                    next_v = min(adj_copy[v])
                    adj_copy[v].discard(next_v)
                    adj_copy[next_v].discard(v)
                    stack.append(next_v)
                else:
                    cycle.append(stack.pop())
            cycle.reverse()
            # cycle is v0, v1, ..., vk where vk == v0 (it's a cycle)
            if len(cycle) > 1 and cycle[0] == cycle[-1]:
                cycle = cycle[:-1]  # Now it's a path representation of the cycle
            if len(cycle) >= 2:
                cycles.append(cycle)
        
        return cycles
    
    def decompose_graph(n, edges):
        """Decompose edges into minimum number of simple paths."""
        if not edges:
            return []
        
        adj = build_adj(n, edges)
        
        # Find odd-degree vertices
        odd_verts = sorted([v for v in range(n) if len(adj[v]) % 2 == 1])
        
        if not odd_verts:
            # All even: single Eulerian circuit
            cycles = eulerian_decompose(adj, n)
            # All edges should be in one cycle if connected, but could be multiple
            # Each cycle becomes a path
            return cycles
        
        # Pair up odd vertices using minimum weight matching
        m = len(odd_verts)
        
        # Compute all-pairs shortest paths among odd vertices
        dist_matrix = [[0]*m for _ in range(m)]
        path_matrix = [[None]*m for _ in range(m)]
        
        for i in range(m):
            dist = bfs_dist(adj, odd_verts[i], n)
            for j in range(m):
                if j != i:
                    if odd_verts[j] in dist:
                        dist_matrix[i][j] = dist[odd_verts[j]]
                    else:
                        dist_matrix[i][j] = float('inf')
        
        # DP for minimum weight perfect matching
        if m <= 20:
            dp = [float('inf')] * (1 << m)
            dp[0] = 0
            choice = [None] * (1 << m)
            
            for mask in range(1, 1 << m):
                # Find first set bit
                first = -1
                for i in range(m):
                    if mask & (1 << i):
                        first = i
                        break
                if first == -1:
                    continue
                for j in range(first + 1, m):
                    if mask & (1 << j):
                        prev = mask ^ (1 << first) ^ (1 << j)
                        cost = dist_matrix[first][j]
                        if cost < float('inf') and dp[prev] + cost < dp[mask]:
                            dp[mask] = dp[prev] + cost
                            choice[mask] = (first, j)
            
            # Reconstruct matching
            matching = []
            mask = (1 << m) - 1
            while mask:
                pair = choice[mask]
                if pair is None:
                    break
                i, j = pair
                matching.append((odd_verts[i], odd_verts[j]))
                mask ^= (1 << i) ^ (1 << j)
        
        else:
            # Greedy fallback
            matching = []
            used = [False] * m
            for i in range(m):
                if used[i]:
                    continue
                best_j = -1
                best_d = float('inf')
                for j in range(i+1, m):
                    if not used[j] and dist_matrix[i][j] < best_d:
                        best_d = dist_matrix[i][j]
                        best_j = j
                if best_j != -1:
                    matching.append((odd_verts[i], odd_verts[best_j]))
                    used[i] = True
                    used[best_j] = True
        
        # Now find edge-disjoint paths for each matched pair
        # Use the original graph, find paths one by one, removing edges as we go
        working_adj = build_adj(n, edges)
        paths = []
        
        for u, v in matching:
            path = bfs_path(working_adj, u, v, n)
            if path is None:
                # Should not happen in connected graph
                path = [u, v] if u in working_adj.get(v, set()) else [u]
            paths.append(path)
            # Remove edges from working_adj
            for k in range(len(path) - 1):
                a, b = path[k], path[k+1]
                working_adj[a].discard(b)
                working_adj[b].discard(a)
        
        # Remaining edges form an Eulerian graph
        remaining_cycles = eulerian_decompose(working_adj, n)
        paths.extend(remaining_cycles)
        
        return paths
    
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    for spec in specs:
        if time.time() > deadline:
            break
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose_graph(n, p if False else p, edges) if False else decompose_graph(n, edges)
        results.append(paths)
    
    # Pad if needed
    for i, spec in enumerate(specs):
        if i < len(results):
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        results.append([[list(e)] for e in edges])
    
    return results
