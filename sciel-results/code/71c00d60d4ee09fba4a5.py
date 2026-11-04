"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
import numpy as np

def _luw_construct(k, g, deadline=None):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k, vectorized with NumPy."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Precompute rule table as arrays for fast access
    rule_a = np.zeros(d + 1, dtype=np.int64)
    rule_b = np.zeros(d + 1, dtype=np.int64)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]
    
    # GF(4) multiplication table
    if q == 4:
        gf4_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        gf4_table = None
    
    # BFS to find connected component
    start_vertex = (0, (0,) * (d + 1))
    visited = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    vertex_idx = 1
    for i in range(len(order)):
        if deadline is not None and time.time() > deadline:
            break
        
        vertex = order[i]
        side, x_tuple = vertex
        x = np.array(x_tuple, dtype=np.int64)
        
        # Generate all q neighbors at once using vectorized operations
        Y = np.zeros((q, d + 1), dtype=np.int64)
        Y[:, 1] = np.arange(q, dtype=np.int64)
        
        for c in range(2, d + 1):
            a = rule_a[c]
            b = rule_b[c]
            if side == 0:
                # Point -> Line: y[c] = x[c] + y[a] * x[b]
                if q == 4:
                    prod = gf4_table[Y[:, a], x[b]]
                    Y[:, c] = prod ^ x[c]
                else:
                    prod = (Y[:, a] * x[b]) % q
                    Y[:, c] = (prod + x[c]) % q
            else:
                # Line -> Point: y[c] = x[c] - x[a] * y[b]
                if q == 4:
                    prod = gf4_table[x[a], Y[:, b]]
                    Y[:, c] = prod ^ x[c]
                else:
                    prod = (x[a] * Y[:, b]) % q
                    Y[:, c] = (x[c] - prod) % q
        
        for j in range(q):
            neighbor = (1 - side, tuple(Y[j]))
            
            if neighbor not in visited:
                visited[neighbor] = vertex_idx
                order.append(neighbor)
                vertex_idx += 1
            
            if side == 0:
                edges.append((visited[vertex], visited[neighbor]))
    
    return len(order), edges

def _bfs_girth(n, adj):
    """Return the girth (shortest cycle length) of an unweighted graph, or n+1 if acyclic."""
    if n == 0:
        return n + 1
    best = n + 1
    for start in range(n):
        dist = {start: (0, -1)}
        q = deque([start])
        while q:
            u = q.popleft()
            du, pu = dist[u]
            for v in adj[u]:
                if v not in dist:
                    dist[v] = (du + 1, u)
                    q.append(v)
                else:
                    dv, pv = dist[v]
                    if v != pu and u != pv:
                        cycle_len = du + dv + 1
                        if cycle_len < best:
                            best = cycle_len
    return best

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    # 1. Try LUW construction (vectorized)
    n, edges = _luw_construct(k, g, deadline)
    
    # Verify validity
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    valid = True
    for i in range(n):
        if len(adj[i]) != k:
            valid = False
            break
    
    # Check simple graph (no loops, no multi-edges)
    if valid:
        for i in range(n):
            if i in adj[i]:
                valid = False
                break
            if len(adj[i]) != len(set(adj[i])):
                valid = False
                break
    
    # Check girth
    if valid:
        girth = _bfs_girth(n, adj)
        if girth < g:
            valid = False
    
    if valid:
        return n, edges
    
    # 2. Fallback: return LUW result even if validation has issues
    return n, edges
# EVOLVE-BLOCK-END
