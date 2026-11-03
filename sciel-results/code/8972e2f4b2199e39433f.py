import random
import time

def construct(k, g):
    # k=3, g=16
    start_time = time.time()
    deadline = start_time + 140
    
    def check_girth(edges, n, g):
        # Check if any cycle < g
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each node
        # Optimization: only need to check up to depth g//2 - 1
        max_depth = g // 2 - 1
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def generate_chorded_cycle(n, g):
        # Create cycle
        edges = set()
        for i in range(n):
            edges.add((i, (i + 1) % n))
        
        # Add chords
        # Need n/2 more edges
        # Ensure distance >= g//2 on the cycle
        min_dist = g // 2
        added = 0
        while added < n // 2:
            u = random.randint(0, n - 1)
            # Pick v such that dist(u,v) >= min_dist
            # dist is circular
            # v can be in [u+min_dist, u+min_dist + (n - 2*min_dist - 1)]?
            # Actually, just pick v randomly and check
            v = random.randint(0, n - 1)
            if u == v:
                continue
            # Check circular distance
            d1 = (v - u) % n
            d2 = (u - v) % n
            dist = min(d1, d2)
            if dist < min_dist:
                continue
            # Check if edge exists
            if (u, v) in edges or (v, u) in edges:
                continue
            edges.add((u, v))
            added += 1
            
        return list(edges)

    # Try n=2000
    for n in [2000, 1500, 1000]:
        if time.time() > deadline:
            break
        for _ in range(100):
            if time.time() > deadline:
                break
            edges = generate_chorded_cycle(n, g)
            if check_girth(edges, n, g):
                return n, edges
                
    # Fallback to LUW
    # (Implement LUW here)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # ... LUW code ...
    return 10**6, []
