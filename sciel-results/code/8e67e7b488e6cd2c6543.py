"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    deadline = time.time() + 80
    
    def verify(edges_to_add):
        """Verify: no duplicates, triangle-free, connected, diameter <= 4"""
        edge_set = set(base_edges)
        for u, v in edges_to_add:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Triangle-free check
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
        
        # BFS for diameter
        def bfs(start):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            idx = 0
            while idx < len(queue):
                u = queue[idx]
                idx += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            return dist
        
        max_d = 0
        for i in range(n):
            ds = bfs(i)
            if -1 in ds:
                return False
            max_d = max(max_d, max(ds))
        
        return max_d <= 4
    
    # Single hub approach: pick hub, connect to max independent set of non-neighbors
    def max_independent_set_on_segments(segments):
        """Find max independent set on a collection of path segments"""
        total = 0
        for seg in segments:
            if not seg:
                continue
            length = len(seg)
            # Max independent set of a path of 'length' vertices
            total += (length + 1) // 2
        return total
    
    best_edges = None
    best_count = 999
    
    # Try each possible hub
    for hub in range(n):
        if time.time() > deadline:
            break
        
        # Vertices available to connect to hub (not hub itself, not adjacent in path)
        available = [v for v in range(n) if v != hub and v not in (hub-1, hub+1) and 0 <= v < n]
        
        # Split into segments (contiguous runs)
        segments = []
        current = []
        for v in sorted(available):
            if not current or v == current[-1] + 1:
                current.append(v)
            else:
                if current:
                    segments.append(current)
                current = [v]
        if current:
            segments.append(current)
        
        # Build the independent set: greedily pick from each segment
        edges_to_add = []
        for seg in segments:
            # Pick every other vertex starting from first
            for i in range(0, len(seg), 2):
                edges_to_add.append((hub, seg[i]))
        
        # Verify
        if verify(edges_to_add):
            if len(edges_to_add) < best_count:
                best_count = len(edges_to_add)
                best_edges = edges_to_add
    
    # Try two-hub approach
    for h1 in range(n):
        if time.time() > deadline:
            break
        for h2 in range(h1 + 2, n):
            if time.time() > deadline:
                break
            # h1 and h2 must not be adjacent
            if abs(h1 - h2) == 1:
                continue
            
            edges_to_add = []
            # Connect each vertex to the closer hub if not adjacent
            for v in range(n):
                if v == h1 or v == h2:
                    continue
                d1 = abs(v - h1)
                d2 = abs(v - h2)
                if d1 < d2:
                    if v != h1 - 1 and v != h1 + 1:
                        edges_to_add.append((h1, v))
                elif d2 < d1:
                    if v != h2 - 1 and v != h2 + 1:
                        edges_to_add.append((h2, v))
                else:
                    # Equal distance, connect to h1 if possible
                    if v != h1 - 1 and v != h1 + 1:
                        edges_to_add.append((h1, v))
            
            if verify(edges_to_add):
                if len(edges_to_add) < best_count:
                    best_count = len(edges_to_add)
                    best_edges = edges_to_add
    
    if best_edges is None:
        # Fallback
        hub = 12
        edges_to_add = []
        for v in range(n):
            if v == hub or abs(v - hub) == 1:
                continue
            if (v - hub) % 2 == 0:  # Even distance
                edges_to_add.append((hub, v))
        best_edges = edges_to_add
    
    return [[int(u), int(v)] for u, v in best_edges]
# EVOLVE-BLOCK-END
