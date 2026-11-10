import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 3:
        return _lnw_construct(k, g)
    
    # Try circulant graph search first (structural change: different graph family)
    best_n = float('inf')
    best_edges = None
    best_girth = 0
    
    # Search over odd n for circulant graphs with jumps {1, s, t}
    # n must be odd for a simple cubic circulant with 3 jumps
    # Girth check: no cycle shorter than g
    
    # Lower bound on n from Moore bound for cubic graphs:
    # n >= 2 + 2*(3*(3^(d-1)-1)/2) where g = 2d+2
    # For g=16, d=7, n >= 2 + 2*(3*(3^6-1)/2) = 2 + 3*(729-1) = 2 + 3*728 = 2186
    # Wait, that's the Moore bound. But best known is 936, so the Moore bound is not tight.
    # Actually for cubic graphs, the Moore bound for girth g=2d+2 is:
    # n >= 2 * (3*(3^(d-1)-1)/2) + 2 = 3*(3^(d-1)-1) + 2
    # For d=7: 3*(729-1)+2 = 3*728+2 = 2186
    # But 936 < 2186, so the Moore bound is not achievable. The actual lower bound is 512.
    
    # Let's search n from a reasonable range
    # We'll try n from 100 to 2000, odd only
    min_n_to_try = 100
    max_n_to_try = 2000
    
    for n in range(min_n_to_try, max_n_to_try + 1, 2):  # odd n
        if time.time() > deadline - 30:
            break
        if n >= best_n:
            continue
            
        # Try all pairs (s, t) with 2 <= s < t <= n//2
        # and s != t, and 1 not in {s,t}
        found = False
        for s in range(2, n // 2 + 1):
            if time.time() > deadline - 30:
                break
            for t in range(s + 1, n // 2 + 1):
                if s == t:
                    continue
                # Check that {1, s, t} are distinct mod n
                # and that the graph is simple (no loops, no repeated edges)
                # For circulant with jumps {1, s, t}, edges are (i, i+1), (i, i+s), (i, i+t) mod n
                # This is simple if n > 2*max(s,t) and s != t and s != 1 and t != 1
                
                # Build edges
                edges = []
                for i in range(n):
                    edges.append((i, (i + 1) % n))
                    edges.append((i, (i + s) % n))
                    edges.append((i, (i + t) % n))
                
                # Remove duplicates and self-loops
                edge_set = set()
                clean_edges = []
                for u, v in edges:
                    if u == v:
                        continue
                    key = (min(u, v), max(u, v))
                    if key not in edge_set:
                        edge_set.add(key)
                        clean_edges.append(key)
                
                # Check degree
                deg = [0] * n
                for u, v in clean_edges:
                    deg[u] += 1
                    deg[v] += 1
                if any(d != 3 for d in deg):
                    continue
                
                # Check girth
                adj = _edges_to_adj(clean_edges, n)
                girth = _calculate_girth_fast(adj, n, g)
                
                if girth >= g:
                    return n, clean_edges
                else:
                    if n < best_n:
                        best_n = n
                        best_edges = clean_edges
                        best_girth = girth
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    return _lnw_construct(3, g)

def _edges_to_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _calculate_girth_fast(adj, n, target_girth):
    """
    Returns the girth if < target_girth, or target_girth if girth >= target_girth.
    This allows early termination if we only care about girth >= target.
    """
    min_cycle = target_girth
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= min_cycle - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle == 3: # Minimum possible for simple graph
                                return 3
    return min_cycle

def _lnw_construct_params(k, d):
    q = k
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

def _lnw_construct(k, g):
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)
