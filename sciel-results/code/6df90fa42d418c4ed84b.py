import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 114.0

    def luw_construct(k, g):
        # ... same as before ...
        pass

    if k != 4 or g != 10:
        return luw_construct(k, g)

    # Optimized search for 320
    def fast_girth(adj, upper):
        n = len(adj)
        best = upper
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] * 2 >= best:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif dist[v] < dist[u]:
                        cyc = dist[u] + dist[v] + 1
                        if cyc < best:
                            best = cyc
                            if best <= g:
                                return best
            if best <= g:
                return best
        return best

    def find_320():
        n = 320
        # Try a few fixed seeds to ensure reproducibility and speed
        for seed in [42, 123, 999, 2024]:
            random.seed(seed)
            # Generate random 4-regular
            adj = None
            for _ in range(50):
                # ... generation logic ...
                pass
            
            if adj is None: continue
            
            # Local search
            # ...
            
            if fast_girth(adj, g) >= g:
                return adj
        
        return None

    # Try smaller sizes
    for target_n in [300, 280, 260, 243]:
        if time.time() > deadline - 30: break
        # Try to construct target_n
        # ...
        
    # Fallback to 320
    adj = find_320()
    if adj:
        edges = []
        for u in range(320):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return 320, edges

    return luw_construct(k, g)
