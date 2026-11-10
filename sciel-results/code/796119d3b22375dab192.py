import time
from collections import deque

def construct():
    t0 = time.time()
    deadline = 85.0
    n = 16
    
    # Construct the "Double Star with Bridge" graph
    # Vertices:
    # 0: Center 1 (C1)
    # 1: Center 2 (C2)
    # 2-8: Leaves of C1 (L1)
    # 9-15: Leaves of C2 (L2)
    
    edges = []
    
    # C1 (0) connected to L1 (2-8)
    for i in range(2, 9):
        edges.append([0, i])
        
    # C1 (0) connected to L2 (9-15)
    for i in range(9, 16):
        edges.append([0, i])
        
    # C2 (1) connected to L2 (9-15)
    for i in range(9, 16):
        edges.append([1, i])
        
    # Bridge between C1 (0) and C2 (1)
    edges.append([0, 1])
    
    # Total edges: 7 + 7 + 7 + 1 = 22
    
    if _verify(edges, n):
        return edges
    
    # Fallback: Star (15 edges)
    return [[0, v] for v in range(1, n)]

def _verify(edges, n):
    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def diam():
        maxd = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        q.append(w)
            if -1 in dist:
                return None
            maxd = max(maxd, max(dist))
        return maxd
    
    if diam() != 2:
        return False
    
    for (u, v) in edges:
        adj2 = {x: set() for x in range(n)}
        for x in range(n):
            for y in adj[x]:
                if not (x == u and y == v) and not (x == v and y == u):
                    adj2[x].add(y)
        
        def diam2():
            maxd = 0
            for s in range(n):
                dist = [-1] * n
                dist[s] = 0
                q = deque([s])
                while q:
                    uu = q.popleft()
                    for w in adj2[uu]:
                        if dist[w] == -1:
                            dist[w] = dist[uu] + 1
                            q.append(w)
                if -1 in dist:
                    return None
                maxd = max(maxd, max(dist))
            return maxd
        
        d2 = diam2()
        if d2 is not None and d2 <= 2:
            return False
    
    return True
