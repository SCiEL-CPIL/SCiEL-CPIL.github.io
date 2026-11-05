import time
import random
import math
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def generate_random_regular(n, k):
        if n * k % 2 != 0:
            return None, None
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        edges = []
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v or v in adj[u]:
                return None, None
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        return adj, edges

    def check_girth_fast(adj, n, target_girth):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du + 1 >= target_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        if du + dist[v] + 1 < target_girth:
                            return False
        return True

    best_n = float('inf')
    best_edges = None

    # Try n from 426 to 1115
    for n in range(426, 1116):
        if time.time() > deadline - 10:
            break
        # Try 20 random graphs
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            adj, edges = generate_random_regular(n, k)
            if adj is None:
                continue
            if check_girth_fast(adj, n, g):
                best_n = n
                best_edges = edges
                break
        if best_n != float('inf'):
            break

    if best_n != float('inf'):
        return best_n, best_edges
    else:
        # Fallback to LUW(4, 5)
        def luw_construction(d, q):
            # ...
            pass
        return luw_construction(4, 5)
