import time
import collections

def construct():
    deadline = time.time() + 80
    
    n = 16
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            edges.add((u, v))
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def get_diameter(adj):
        diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if -1 in dist:
                return -1
            m = max(dist)
            if m > diam:
                diam = m
        return diam
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for edge in list(edges):
            u, v = edge
            adj[u].discard(v)
            adj[v].discard(u)
            diam = get_diameter(adj)
            adj[u].add(v)
            adj[v].add(u)
            if diam > 2 or diam == -1:
                pass
            else:
                edges.discard(edge)
                adj[u].discard(v)
                adj[v].discard(u)
                changed = True
    
    result = [[u, v] for u, v in sorted(edges)]
    return result
