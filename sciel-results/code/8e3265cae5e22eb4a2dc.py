import time
from collections import deque
import numpy as np
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k == 3 and g == 16:
        # Try circulant-based construction:
        # A 3-regular circulant graph on n vertices has edges {i, i+a, i-b} mod n
        # for some offsets a, b with a != b, a != -b, etc.
        # We search over n and offset pairs to find one with girth >= g.
        
        best_n = float('inf')
        best_edges = None
        
        # Moore bound for k=3, g=16: 2 + 2*(3*(3^(7)-1)/2) = 2 + 3*(3^8-1) = 2 + 3*6560 = 19682... 
        # Wait, Moore bound for cubic: n >= 2 + k*(k-1)^(g/2 - 1) * ... 
        # Actually for girth g (even), Moore bound: n >= 2*(k-1)^(g/2) - 2*(k-1)^(g/2 - 1)
        # For k=3, g=16: n >= 2*2^8 - 2*2^7 = 512 - 256 = 256? No...
        # Standard Moore bound for cubic girth 2g: n >= 2*(2^(g-1) - 1) = 2*2^(g-1) - 2
        # For girth 16 = 2*8: n >= 2*(2^8 - 1) = 510? That doesn't match the stated 512.
        # Let me just search from reasonable sizes.
        
        # Search n from 256 upward, trying various offset combinations
        # For a 3-regular circulant on n vertices, we need offsets a, b such that
        # the graph is simple and 3-regular. The neighbors of i are i+a, i-a, i+b, i-b mod n.
        # That's 4 neighbors if a != b. For exactly 3, we need one of the offsets to be n/2 (so i+n/2 = i-n/2).
        # So one offset must be n/2 (for even n), giving edges to i+n/2, and two other offsets a, b.
        
        # Alternative: use a "blow-up" or other construction.
        # Let me try the LUW construction as primary and add a systematic circulant search.
        
        # First, try LUW which is known to work
        n_luw, edges_luw = _luw_graph(k, g)
        best_n = n_luw
        best_edges = edges_luw
        
        # Now try to do better with circulant graphs
        # For 3-regular circulant: n must be even, one offset is n/2
        # Neighbors of i: (i + n//2) % n, (i + a) % n, (i - a) % n, (i + b) % n, (i - b) % n
        # That's 5 neighbors unless a == b or a == n/2 or b == n/2 or a == -b, etc.
        # For exactly 3 neighbors: we need the set {i+n/2, i+a, i-a, i+b, i-b} to have size 3.
        # This happens if a == b (giving 3 distinct: i+n/2, i+a, i-a) or a == n/2 (giving i+a = i-n/2 = i+n/2, so 3 distinct: i+a, i+b, i-b).
        
        # Case 1: offsets are {n//2, a, -a} where a != n//2, a != 0
        # Neighbors: i+n/2, i+a, i-a. That's 3 neighbors if a != n/2 and a != 0 and 2a != 0 mod n.
        
        # Let's search over n and a
        for n in range(256, 2000, 2):  # even n only
            if time.time() > deadline - 15:
                break
            if n >= best_n:
                continue
            
            half = n // 2
            # Try various a values
            for a in range(1, half):
                if 2 * a % n == 0:
                    continue  # i+a == i-a, not 3-regular
                if time.time() > deadline - 15:
                    break
                
                # Check girth using BFS
                if _check_girth_circulant(n, a, half, g):
                    if n < best_n:
                        best_n = n
                        # Build edges
                        edges = []
                        for i in range(n):
                            for nb in [(i + a) % n, (i - a) % n, (i + half) % n]:
                                if i < nb:
                                    edges.append((i, nb))
                        best_edges = edges
                        break
        
        if best_edges is not None:
            return best_n, best_edges
        
        return n_luw, edges_luw
    
    # General fallback
    n_luw, edges_luw = _luw_graph(k, g)
    return n_luw, edges_luw

def _check_girth_circulant(n, a, half, g_limit):
    """Check if the 3-regular circulant graph on n vertices with offsets {a, -a, n/2} has girth >= g_limit."""
    # BFS from vertex 0, tracking shortest path
    # For circulant graphs, we can be smart but let's just do BFS
    # Neighbors of i: (i+a)%n, (i-a)%n, (i+half)%n
    
    # We need to check all vertices, but by symmetry we only need vertex 0
    # Actually no, girth is the minimum cycle length, so we need to check from each vertex
    # But for circulant graphs, the girth can be found by checking from vertex 0
    # since the graph is vertex-transitive.
    
    # BFS from vertex 0
    dist = [-1] * n
    dist[0] = 0
    queue = deque([0])
    
    while queue:
        u = queue.popleft()
        if dist[u] >= g_limit - 1:
            continue
        for v in [(u + a) % n, (u - a) % n, (u + half) % n]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
            else:
                # Found a cycle of length dist[u] + dist[v] + 1
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < g_limit:
                    return False
    return True

def _check_girth_fast(adj, n, g_limit):
    # Use BFS from each vertex to check for cycles shorter than g_limit
    # For a 3-regular graph, we can use a more efficient approach
    # by checking non-backtracking walks
    max_depth = (g_limit - 1) // 2
    
    for start in range(n):
        # dist[v] = shortest distance from start to v
        # parent[v] = parent of v in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < max_depth:
                        queue.append(v)
                else:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g_limit:
                        return False
    return True

def _luw_graph(k, g):
    # Lazebnik-Ustimenko-Woldar construction for 3-regular graphs
    # This produces a 3-regular graph with girth at least g
    # Same as before
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
