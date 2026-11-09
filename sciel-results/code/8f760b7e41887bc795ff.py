import time
import random

def construct(k, g):
    def luw_construct_original(k, g):
        # ... (copy from current) ...
        pass

    # New random search
    def generate_random_cubic(n):
        if n % 2 != 0: return None
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        random.shuffle(stubs)
        edges = []
        seen = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v: continue
            if (u, v) in seen or (v, u) in seen: continue
            edges.append((u, v))
            seen.add((u, v))
            seen.add((v, u))
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg): return None
        return edges

    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    if k == 3:
        start_time = time.time()
        for n in range(512, 600, 2):
            if time.time() - start_time > 100: break
            for _ in range(50):
                edges = generate_random_cubic(n)
                if edges and check_girth(n, edges, g):
                    return n, edges
        
        # Fallback to original
        return luw_construct_original(k, g)
        
    return 0, []
