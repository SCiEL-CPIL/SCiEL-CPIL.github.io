"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Strategy:
    1. Try Greedy Random Construction for n=1116 (best known).
    2. Fallback to LZW Algebraic Construction (guaranteed valid, larger n).
    """
    deadline = time.time() + 100  # Safety margin for 120s limit
    
    # Best known for k=5, g=9 is 1116.
    # We try to find a graph with n=1116.
    # If it fails, we try slightly larger n.
    # If all fail, fallback to LZW.
    
    target_ns = [1116, 1200, 1300]
    
    for n in target_ns:
        if time.time() > deadline:
            break
        result = _greedy_construct(k, g, n, deadline)
        if result is not None:
            return result
            
    # Fallback to LZW
    return _lzw_construct(k, g)

def _greedy_construct(k, g, n, deadline):
    """
    Greedy construction of a k-regular graph with girth >= g.
    """
    max_restarts = 20  # Increased restarts
    for _ in range(max_restarts):
        if time.time() > deadline:
            return None
        
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        edges_added = 0
        target_edges = n * k // 2
        
        active_vertices = list(range(n))
        random.shuffle(active_vertices)
        
        # Pre-compute valid neighbors is expensive, do it on the fly
        # But we can use a "forbidden" set based on BFS
        
        while edges_added < target_edges:
            if time.time() > deadline:
                return None
            
            # Find vertices with deg < k
            candidates_u = [u for u in active_vertices if deg[u] < k]
            if not candidates_u:
                break
            
            # Pick vertex with fewest valid candidates
            best_u = None
            best_valid_vs = None
            best_count = float('inf')
            
            # Sample to speed up
            sample_size = min(len(candidates_u), 50)
            sampled = random.sample(candidates_u, sample_size)
            
            for u in sampled:
                valid_vs = _find_valid_neighbors(adj, deg, u, k, g, n)
                if len(valid_vs) < best_count:
                    best_count = len(valid_vs)
                    best_u = u
                    best_valid_vs = valid_vs
                    if best_count == 0:
                        break
            
            if best_u is None or len(best_valid_vs) == 0:
                return None
            
            v = random.choice(best_valid_vs)
            
            adj[best_u].add(v)
            adj[v].add(best_u)
            deg[best_u] += 1
            deg[v] += 1
            edges.append((best_u, v))
            edges_added += 1
            
        if all(d == k for d in deg):
            if _verify_girth(n, edges, k, g):
                return (n, edges)
                
    return None

def _find_valid_neighbors(adj, deg, u, k, g, n):
    """Find all valid neighbors v for u that maintain girth constraint."""
    forbidden = set(adj[u])
    forbidden.add(u)
    
    # BFS from u up to depth g-2
    # Vertices within distance g-2 cannot be neighbors of u
    # because it would create a cycle of length <= (g-2) + 1 + 1? 
    # Wait. If v is at distance d from u, and we add edge (u,v), 
    # the cycle length is d + 1.
    # We need cycle length >= g.
    # So d + 1 >= g => d >= g - 1.
    # So v must be at distance >= g - 1 from u.
    # Thus, vertices at distance < g - 1 are forbidden.
    # Max depth to forbid is g - 2.
    
    max_depth = g - 2
    dist = {u: 0}
    queue = deque([(u, 0)])
    
    while queue:
        node, d = queue.popleft()
        if d >= max_depth:
            continue
        for nb in adj[node]:
            if nb not in dist:
                dist[nb] = d + 1
                if d + 1 <= max_depth:
                    queue.append((nb, d + 1))
    
    valid_vs = []
    for v in range(n):
        if v in dist or v in forbidden:
            continue
        if deg[v] < k:
            valid_vs.append(v)
    
    return valid_vs

def _verify_girth(n, edges, k, g):
    """
    Verify the graph has girth >= g and is k-regular.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # BFS from each vertex to find shortest cycle
    # Optimization: If we find a cycle of length < g, return False early.
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < g:
                        return False
                        
    return True

def _lzw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph"""
    q = k
    # For girth >= 9, we need d such that 2d >= 9 => d >= 4.5 => d=5.
    # If d=4, girth is 8 (usually). So d=5 is safe.
    d = 5
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
