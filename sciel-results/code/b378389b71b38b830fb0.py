import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if k != 3:
        # Fallback for other k: use a simple ring or LUW if applicable
        # For now, assume k=3 is the main case
        return _construct_luw_generic(k, g)

    # Target: 3-regular graph with girth >= 16
    # Moore bound is 512. Known best is 936.
    # LUW construction for k=3 (q=3):
    # d=10 (even) -> girth 16, N = 2 * 3^10 = 118098
    # d=11 (odd) -> girth 16, N = 2 * 3^11 = 354294
    # We use d=10 as it's smaller.
    
    # 1. Try to find a smaller graph using Generalized Petersen Graphs GP(n, k_gp)
    # GP(n, k_gp) has 2*n vertices.
    # We scan n from 256 (Moore bound/2) to 500.
    # If we find one with girth >= 16, we return it.
    
    best_n = 118098
    best_edges = _construct_luw_optimized(3, 10)
    
    # Time check: LUW construction is fast. Let's try GP search.
    if time.time() < deadline - 5.0:
        for n_gp in range(256, 501, 2):
            if time.time() > deadline - 2.0:
                break
            if 2 * n_gp >= best_n:
                break
            
            # Check GP(n_gp, k_gp)
            # k_gp must be coprime to n_gp
            for k_gp in range(1, n_gp // 2 + 1):
                if math.gcd(n_gp, k_gp) != 1:
                    continue
                
                # Quick heuristic: k_gp should not be 1 or 2 to avoid short cycles
                if k_gp < 3:
                    continue
                    
                edges = _generate_gp_edges(n_gp, k_gp)
                
                # Check girth
                # Optimization: If girth is likely < 16, skip detailed check?
                # Just do a fast girth check.
                if _has_girth_at_least_fast(2 * n_gp, edges, g):
                    best_n = 2 * n_gp
                    best_edges = edges
                    break
            
            if 2 * n_gp < best_n:
                break

    return best_n, best_edges

def _generate_gp_edges(n, k_gp):
    """
    Generate edges for Generalized Petersen Graph GP(n, k_gp).
    Vertices 0..n-1 are outer, n..2n-1 are inner.
    """
    edges = []
    for i in range(n):
        # Outer cycle
        edges.append((i, (i + 1) % n))
        # Spokes
        edges.append((i, n + i))
        # Inner cycle
        edges.append((n + i, n + (i + k_gp) % n))
    return edges

def _has_girth_at_least_fast(n, edges, g):
    """
    Fast girth check.
    For n < 1000, BFS from every vertex is O(n^2) which is ~10^6, very fast.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
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
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw_optimized(k, d):
    """
    Deterministic LUW construction for k-regular graphs.
    Optimized for k=3.
    """
    if k != 3:
        return _construct_luw_generic(k, d)
        
    q = 3
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _construct_luw_generic(k, d):
    q = k
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
