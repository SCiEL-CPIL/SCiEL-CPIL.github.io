"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def time_left():
        return deadline - time.time()

    # Moore bound for lower bound on n
    if g % 2 == 0:
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        if k == 2:
            moore = g
        else:
            h = (g - 1) // 2
            moore = 1 + k * ((k - 1) ** h - 1) // (k - 2) if k > 2 else g

    best_n = None
    best_edges = None

    def verify_girth(n, edges, min_girth):
        """Verify no cycle shorter than min_girth using BFS from each vertex."""
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check simple: no self-loops, no duplicate edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        # BFS from each vertex to find shortest cycle
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def try_explicit_construction(d_target):
        """
        The Lazebnik-Ustimenko-Woldar construction.
        We try to use a specific depth d_target.
        """
        q = k # The seed uses q=k. For k=5, q=5.
        
        # Field operations
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q

        # Rule definition
        # The seed defines rules up to d.
        # We need to ensure rules are defined for all c in 2..d_target
        rule = {}
        if d_target >= 2:
            rule[2] = (1, 1)
        if d_target >= 3:
            rule[3] = (2, 1)
        if d_target >= 4:
            rule[4] = (1, 2)
        for c in range(5, d_target + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                # x is a tuple of length d_target + 1?
                # In seed: y = [None, first] + [0] * (d - 1)
                # This creates a list of length 1 + 1 + d - 1 = d + 1
                # Indices: 0 is None, 1 is first, 2..d are computed
                y = [None, first] + [0] * (d_target - 1)
                for c in range(2, d_target + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_target)
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

    # Strategy: Try smaller depths first to get smaller n.
    # For k=5, g=9:
    # d=4 -> n = 2 * 5^4 = 1250. Girth might be 8 or 9.
    # d=5 -> n = 2 * 5^5 = 6250. Girth >= 10.
    
    # We will try d values in increasing order of n (which is increasing d).
    # But we need girth >= g.
    # Let's try d=4 first.
    
    # Determine max d to try. 
    # n grows as 2 * k^d. 
    # We should stop if n becomes too large or time runs out.
    
    # For k=5, d=4 is 1250. d=5 is 6250.
    # Let's try d=4.
    
    candidate_ds = []
    # We want to find the smallest n.
    # n is proportional to k^d.
    # So we try small d first.
    
    # However, small d might have small girth.
    # We verify girth.
    
    # Let's try d from 3 up to some limit.
    # For k=5, g=9, d=3 gives n=250. Girth likely < 9.
    # d=4 gives n=1250.
    
    for d in range(3, 8): # Try d=3,4,5,6,7
        if time_left() < 10:
            break
            
        n_try, edges_try = try_explicit_construction(d)
        
        # Optional: Quick check on n size. 
        # If n is already larger than a known good solution, we might skip?
        # But we don't know a good solution here other than the seed's.
        # The seed likely used d=5.
        # So if we find a valid graph with d=4, it's better.
        
        if verify_girth(n_try, edges_try, g):
            if best_n is None or n_try < best_n:
                best_n = n_try
                best_edges = edges_try
            # If we found a valid graph with small d, we can stop?
            # Since n increases with d, smaller d is better.
            # So if d=3 works, it's better than d=4.
            # So we break after the first valid one?
            # Yes, because we iterate d in increasing order.
            break

    if best_n is None:
        # Fallback to the seed's logic if nothing found (should not happen for d=5)
        # The seed used d=5 for g=9.
        d_seed = 5
        if (g - 5) % 2 == 1:
            d_seed = g - 5
        else:
            d_seed = g - 4
        if d_seed < 3: d_seed = 3
        
        n_exp, edges_exp = try_explicit_construction(d_seed)
        return n_exp, edges_exp

    return best_n, best_edges
# EVOLVE-BLOCK-END
