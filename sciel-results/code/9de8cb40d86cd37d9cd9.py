"""Triangle-free diameter-four augmentation"""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _valid_fast(added_edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        for i in range(len(queue)):
            u = queue[i]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist or max(dist) > 4:
            return False
    return True

def construct():
    deadline = time.time() + 85
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            if (u % 2) == (v % 2):
                continue
            candidates.append((u, v))
    candidates.sort(key=lambda e: -abs(e[0] - e[1]))
    
    for k in range(3, 8):
        if time.time() > deadline:
            break
        top_size = 25 if k >= 5 else 30
        top = candidates[:top_size]
        for combo in combinations(top, k):
            if time.time() > deadline:
                break
            if _valid_fast(list(combo)):
                return [[u, v] for u, v in combo]
    
    fb = [[0, 23], [1, 22], [2, 21], [3, 20], [4, 19]]
    if _valid_fast(fb):
        return fb
    return []

# EVOLVE-BLOCK-END
