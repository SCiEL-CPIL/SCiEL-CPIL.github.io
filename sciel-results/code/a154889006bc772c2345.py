import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    n = 16
    all_edges = [[i, j] for i in range(n) for j in range(i+1, n)]
    
    # Fallback: Star (15 edges)
    best_edges = [[0, i] for i in range(1, n)]
    
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
            
        edges = all_edges[:]
        random.shuffle(edges)
        
        i = 0
        while i < len(edges):
            u, v = edges[i]
            if _is_essential(edges, n, u, v):
                i += 1
            else:
                edges.pop(i)
        
        if len(edges) > len(best_edges):
            best_edges = edges[:]
            
    return best_edges

def _is_essential(edges, n, u, v):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        if (a == u and b == v) or (a == v and b == u):
            continue
        adj[a].add(b)
        adj[b].add(a)
        
    # Check Connectivity
    visited = [False] * n
    visited[0] = True
    q = [0]
    head = 0
    while head < len(q):
        x = q[head]
        head += 1
        for y in adj[x]:
            if not visited[y]:
                visited[y] = True
                q.append(y)
    if not all(visited):
        return True
        
    # Check Diameter
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            x = q[head]
            head += 1
            for y in adj[x]:
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    q.append(y)
        if max(dist) > 2:
            return True
            
    return False
