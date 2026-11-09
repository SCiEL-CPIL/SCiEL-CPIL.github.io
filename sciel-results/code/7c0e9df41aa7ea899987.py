import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 115 second deadline (safety margin)
    
    # L-U-W construction (optimized)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Precompute multiplication table for q=4
    if q == 4:
        mul_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int32)
    else:
        mul_table = None
    
    # Build rule: rule[c] = (a, b) for c in 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # BFS-based construction
    index = {}
    order = []
    edges = []
    
    # Start vertex: (0, (0,)*d)
    start_key = (0, tuple([0]*d))
    index[start_key] = 0
    order.append(start_key)
    
    # BFS to build graph
    queue = deque([0])
    
    while queue:
        if time.time() > deadline - 15:
            break
        
        idx = queue.popleft()
        side, x = order[idx]
        
        for first in range(q):
            y = [0] * (d + 1)  # y[0] is dummy, y[1..d]
            y[1] = first
            
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # y[c] = x[c] + y[a] * x[b] (in the appropriate group)
                    if q == 4:
                        y[c] = x[c] ^ mul_table[y[a], x[b]]
                    else:
                        y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    # y[c] = x[c] - x[a] * y[b]
                    if q == 4:
                        y[c] = x[c] ^ mul_table[x[a], y[b]]
                    else:
                        y[c] = (x[c] - x[a] * y[b]) % q
            
            new_side = 1 - side
            new_key = (new_side, tuple(y[1:]))
            
            if new_key not in index:
                index[new_key] = len(order)
                order.append(new_key)
                queue.append(len(order) - 1)
            
            if side == 0:
                edges.append((idx, index[new_key]))
    
    n = len(order)
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Verify girth >= g using BFS
    girth_ok = True
    for start in range(n):
        if time.time() > deadline - 5:
            break
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q_bfs = deque([start])
        
        while q_bfs:
            u = q_bfs.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q_bfs.append(v)
                elif v != parent[u] and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        girth_ok = False
                        break
            if not girth_ok:
                break
        if not girth_ok:
            break
    
    return n, edges
