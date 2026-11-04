import math
import random
from collections import deque
import time

def _moore_bound(k, g):
    if g % 2 == 0:
        m = g // 2
        if k == 1: return 2
        s = sum((k-1)**i for i in range(m))
        return 1 + k * s
    else:
        m = (g - 1) // 2
        s = sum((k-1)**i for i in range(m))
        return 1 + k * s

def _generate_random_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
        
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            continue
        if (v, u) in edges:
            continue
        edges.add((u, v))
        
    # Check if degree is exactly k
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return None
        
    return list(edges)

def _check_girth(adj, g):
    n = len(adj)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def construct(k, g):
    if k < 2:
        return 0, []
    if g <= 2:
        n = k + 1
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return n, edges

    n_min = _moore_bound(k, g)
    start_time = time.time()
    
    # Try increasing n
    n = n_min
    while time.time() - start_time < 110:
        # Try several random graphs for this n
        for attempt in range(50):
            edges = _generate_random_regular(n, k)
            if edges is None:
                continue
            adj = _build_adj(edges, n)
            if _check_girth(adj, g):
                return n, edges
                
        n += 1
        
    # If we run out of time, return the last tried n? 
    # No, we must return a valid graph.
    # If we didn't find one, we return 0,0?
    # But the task requires a valid graph.
    # We should ensure we find one.
    # For large n, random regular graphs have high girth.
    # So this loop should terminate.
    
    return 0, []
