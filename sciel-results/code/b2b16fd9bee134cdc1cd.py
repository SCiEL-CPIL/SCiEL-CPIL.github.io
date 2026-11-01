import time
import random
from collections import deque

def construct(k, g):
    start = time.time()
    deadline = start + 115.0
    
    def time_left():
        return deadline - time.time()

    def moore_bound(k, g):
        r = g // 2
        return 1 + k * sum((k-1)**i for i in range(r))

    def check_girth(adj, n, g):
        max_depth = g // 2
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                v = q.popleft()
                d = dist[v]
                if d >= max_depth:
                    continue
                for w in adj[v]:
                    if w == parent[v]:
                        continue
                    if dist[w] == -1:
                        dist[w] = d + 1
                        parent[w] = v
                        q.append(w)
                    else:
                        cycle_len = d + 1 + dist[w]
                        if cycle_len < g:
                            return False
        return True

    def generate_random_k_regular(n, k):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != len(set(adj[i])):
                return None
        return adj

    min_n = moore_bound(k, g)
    max_n = min_n + 100 # Search range
    
    for n in range(min_n, max_n + 1):
        if time_left() < 2.0:
            break
        start_n = time.time()
        while time.time() - start_n < 2.0:
            if time_left() < 1.0:
                break
            adj = generate_random_k_regular(n, k)
            if adj is not None:
                if check_girth(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
    return 0, []
