import time
import math
import random
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    For k=4, g=10, it searches for a small circulant graph.
    Falls back to Lazebnik-Ustimenko-Woldar construction if search fails or takes too long.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin
    
    if k != 4 or g != 10:
        return _luw_construct(k, g)

    # Search for circulant graphs C_n(a, b)
    # Known cage is 320. We search up to 400 to be safe, but prioritize small n.
    # If we find a valid circulant graph with n < luw_n, we return it.
    
    # First, get the LUW result to know the baseline
    luw_n, luw_edges = _luw_construct(k, g)
    best_n = luw_n
    best_edges = luw_edges
    
    # Search range: from Moore bound (243) to best_n - 1
    # We don't need to search beyond best_n because we want to improve.
    # However, if no circulant < best_n exists, we return best_n.
    
    max_search_n = min(500, best_n - 1)
    
    for n in range(243, max_search_n + 1):
        if time.time() > deadline:
            break
            
        # Iterate over possible generators a, b
        # a, b in [1, n//2]
        # a != b, a+b != n, 2a != 0, 2b != 0
        for a in range(1, n // 2 + 1):
            if (2 * a) % n == 0:
                continue
            for b in range(a + 1, n // 2 + 1):
                if (2 * b) % n == 0:
                    continue
                if a + b == n:
                    continue
                
                # Quick check: gcd(n, a, b) must be 1 for connectivity
                if math.gcd(math.gcd(n, a), b) != 1:
                    continue
                
                # Check girth
                if _check_circulant_girth(n, a, b, g):
                    # Construct edges
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + a) % n))
                        edges.append((i, (i + b) % n))
                    
                    # Verify exact girth just in case (filter might have false negatives? No, filter is sufficient for rejection, but let's be safe)
                    if _verify_girth(n, edges, g):
                        return n, edges

    # If no circulant found, return LUW
    return best_n, best_edges

def _check_circulant_girth(n, a, b, g):
    """
    Fast filter for girth of circulant graph C_n(a,b).
    Returns True if girth >= g, False otherwise.
    Uses BFS from 0. If any edge (u,v) has dist[u]+dist[v]+1 < g, returns False.
    This is a necessary condition for girth >= g? 
    Actually, if girth < g, there exists a cycle of length L < g.
    Does there always exist an edge (u,v) in that cycle such that dist_0(u)+dist_0(v)+1 < g?
    Yes, as argued before.
    So if this check passes (no such edge), girth might still be < g?
    Wait. The condition "min_{(u,v)} dist[u]+dist[v]+1 < g" implies girth < g.
    The contrapositive: If girth >= g, then min_{(u,v)} dist[u]+dist[v]+1 >= g.
    So if min < g, girth < g.
    If min >= g, girth >= g?
    No. The min is a lower bound on the girth?
    Let's check the triangle again. 0-1-2-0.
    dist(0)=0, dist(1)=1, dist(2)=1.
    Edge (1,2): 1+1+1=3.
    Edge (0,1): 0+1+1=2.
    Edge (0,2): 0+1+1=2.
    Min is 2. Girth is 3.
    So min < girth.
    If min < g, then girth < g?
    If min = 2 and g = 3, min < g. Girth = 3. 3 is not < 3.
    So "min < g" does NOT imply "girth < g".
    It implies "girth <= min + something"?
    Actually, the closed walk length is $dist[u]+1+dist[v]$.
    This walk contains a simple cycle of length $\le dist[u]+1+dist[v]$.
    So if $dist[u]+1+dist[v] < g$, then there is a simple cycle of length $< g$.
    So if we find ANY edge with sum < g, we have a short cycle.
    If we find NO such edge, we don't know.
    
    So this is a **filter for REJECTION**.
    If it finds a short cycle, we reject.
    If it doesn't, we must do a full check.
    """
    dist = [-1] * n
    dist[0] = 0
    q = [0]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        du = dist[u]
        if du >= (g - 2) // 2:
            continue
        
        neighbors = [(u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n]
        for v in neighbors:
            if dist[v] == -1:
                dist[v] = du + 1
                if du + 1 <= (g - 2) // 2:
                    q.append(v)
            else:
                # Check if this edge creates a short cycle
                if du + dist[v] + 1 < g:
                    return False
    return True

def _verify_girth(n, edges, g):
    """
    Exact girth check using BFS from every vertex.
    O(n * E). For n=320, E=640, this is ~200k ops. Fast.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            du = dist[u]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(v)
                else:
                    if du + dist[v] + 1 < g:
                        return False
    return True

def _luw_construct(k, g):
    # Original LUW code
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
