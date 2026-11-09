# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def max_flow_residual(capacity, n, source, sink):
        """Find max flow using Edmonds-Karp (BFS-based)."""
        flow = 0
        while True:
            # BFS to find augmenting path
            parent = [-1] * n
            parent[source] = source
            visited = [False] * n
            visited[source] = True
            queue = [source]
            while queue:
                u = queue.pop(0)
                for v in range(n):
                    if not visited[v] and capacity[u][v] > 0:
                        visited[v] = True
                        parent[v] = u
                        if v == sink:
                            break
                        queue.append(v)
                if visited[sink]:
                    break
            if not visited[sink]:
                break
            # Find min capacity on path
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, capacity[u][v])
                v = u
            # Update residual
            v = sink
            while v != source:
                u = parent[v]
                capacity[u][v] -= path_flow
                capacity[v][u] += path_flow
                v = u
            flow += path_flow
        return flow

    def find_edge_disjoint_paths(edges, n, pairs):
        """Find edge-disjoint paths connecting each pair in pairs.
        Returns list of paths (vertex lists) or None if impossible.
        """
        if not pairs:
            return []
        
        m = len(edges)
        # Build edge list with IDs
        edge_list = list(edges)
        
        # For each pair, we need to find a path
        # Use max-flow: create a flow network where each edge has capacity 1
        # Source = super source connected to first vertex of each pair
        # Sink = super sink connected to second vertex of each pair
        # But we need to ensure edge-disjointness across all paths
        
        # Better approach: iterative - find path for each pair using max-flow
        # with previously used edges removed
        
        remaining_edges = set(edge_list)
        paths = []
        
        for a, b in pairs:
            # Build flow network
            # Nodes: 0 to n-1 for original, n for source, n+1 for sink
            num_nodes = n + 2
            src = n
            snk = n + 1
            capacity = [[0] * num_nodes for _ in range(num_nodes)]
            
            # Add edges with capacity 1
            for u, v in remaining_edges:
                capacity[u][v] += 1
                capacity[v][u] += 1
            
            # Connect source to a, b to sink
            capacity[src][a] = 1
            capacity[b][snk] = 1
            
            # Find max flow (should be 1 if path exists)
            flow = max_flow_residual(capacity, num_nodes, src, snk)
            
            if flow < 1:
                return None  # No path found
            
            # Extract path from residual graph
            # Find the path in the original graph
            # We need to trace back using the flow
            # Reconstruct by finding the path in the residual
            
            # Actually, let's just do a BFS on the remaining edges
            from collections import deque
            parent = {a: None}
            queue = deque([a])
            while queue:
                u = queue.popleft()
                if u == b:
                    break
                for v, w in remaining_edges:
                    if (u, v) == tuple(sorted((v, w))) or (u, w) == tuple(sorted((v, w))):
                        pass
                # This is getting complex. Let me use a simpler BFS.
            
            # Simpler: BFS on remaining edges
            adj = {i: [] for i in range(n)}
            for u, v in remaining_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            parent = {a: None}
            queue = deque([a])
            while queue:
                u = queue.popleft()
                if u == b:
                    break
                for nb in adj[u]:
                    if nb not in parent:
                        parent[nb] = u
                        queue.append(nb)
            
            if b not in parent:
                return None
            
            # Reconstruct path
            path = []
            cur = b
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            
            # Remove used edges
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                remaining_edges.discard(e)
            
            paths.append(path)
        
        return paths

    def min_path_cover(edges, n):
        """Find minimum edge-disjoint simple path cover."""
        if not edges:
            return []
        
        # Compute degrees
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        num_paths = max(1, len(odd_verts) // 2)
        
        if num_paths == 1 and len(odd_verts) == 0:
            # All even: find Eulerian circuit
            adj = {i: [] for i in range(n)}
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Hierholzer's
            used = set()
            def eulerian(start):
                stack = [start]
                circuit = []
                while stack:
                    v = stack[-1]
                    found = False
                    for nb in adj[v]:
                        e = tuple(sorted((v, nb)))
                        if e not in used:
                            used.add(e)
                            stack.append(nb)
                            found = True
                            break
                    if not found:
                        circuit.append(stack.pop())
                circuit.reverse()
                return circuit
            
            start = 0
            while not adj[start]:
                start += 1
            circuit = eulerian(start)
            # circuit[0] == circuit[-1]
            return [circuit[:-1]]
        
        # Pair odd vertices - try different pairings
        # First, try simple greedy pairing
        pairs = []
        for i in range(0, len(odd_verts), 2):
            pairs.append((odd_verts[i], odd_verts[i+1]))
        
        paths = find_edge_disjoint_paths(edges, n, pairs)
        if paths is not None and len(paths) == num_paths:
            # Verify all edges covered
            used = set()
            for path in paths:
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i+1])))
                    if e in used or e not in set(edges):
                        return None
                    used.add(e)
            if used == set(edges):
                return paths
        
        # Fallback: greedy longest path
        return greedy_path_cover(edges, n)

    def greedy_path_cover(edges, n):
        """Greedy: repeatedly find longest simple path in remaining graph."""
        if not edges:
            return []
        
        remaining = set(edges)
        paths = []
        
        while remaining:
            adj = {i: set() for i in range(n)}
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            best_path = None
            best_len = 0
            
            # Try starting from each vertex with small degree first
            candidates = sorted([v for v in range(n) if adj[v]], key=lambda v: len(adj[v]))
            
            for start in candidates:
                # DFS to find longest simple path
                def dfs(path, visited):
                    nonlocal best_path, best_len
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = list(path)
                    cur = path[-1]
                    for nb in adj[cur]:
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            dfs(path, visited)
                            path.pop()
                            visited.remove(nb)
                
                dfs([start], {start})
            
            if best_path is None or len(best_path) < 2:
                e = next(iter(remaining))
                best_path = list(e)
            
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                remaining.discard(e)
            
            paths.append(best_path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        
        paths = min_path_cover(edges, n)
        if paths is None:
            paths = greedy_path_cover(edges, n)
        
        # Validate
        valid = True
        used = set()
        edge_set = set(edges)
        for path in paths:
            if len(path) < 2:
                valid = False
                break
            if len(set(path)) != len(path):
                valid = False
                break
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in edge_set or e in used:
                    valid = False
                    break
                used.add(e)
            if not valid:
                break
        
        if valid and used == edge_set:
            results.append(paths)
        else:
            paths = greedy_path_cover(edges, n)
            results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
