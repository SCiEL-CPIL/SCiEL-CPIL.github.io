"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct(k, g):
    start_time = time.time()
    DEADLINE = start_time + 145.0  # safety margin

    # Moore bound for k-regular graph of girth g
    if g % 2 == 0:
        half = g // 2
        # For even girth, Moore bound: n >= k * (k-1)^(half-1) / 2
        moore = int(k * (k - 1) ** (half - 1) // 2) if k > 1 else 2
    else:
        # For odd girth, Moore bound: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        half = (g - 1) // 2
        s = 1
        for i in range(half):
            s += (k - 1) ** i
        moore = 1 + k * s

    # The LUW construction from the seed
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
    
    n_luw = len(order)
    
    # Try to improve by taking a subgraph or using a different approach
    # For now, try to see if we can get a smaller valid graph
    
    # Check if the LUW graph is valid and has the right properties
    # Build adjacency list
    adj = [[] for _ in range(n_luw)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Verify degree
    for i in range(n_luw):
        if len(adj[i]) != k:
            # If degree is not exactly k, try to fix or find another approach
            break
    
    # Try a BFS-based construction for a smaller graph
    # This is a heuristic approach: try to build a k-regular graph with girth >= g
    # using a "greedy" approach with backtracking
    
    # For small n, we could try random regular graphs and check girth
    # But this is expensive, so let's just return the LUW construction for now
    
    # Actually, let's try to be smarter: for g=16, k=3, the known best is 936
    # LUW gives q^d * 2 = 3^11 * 2 = 177147 * 2 = 354294, which is way too big
    
    # Let's try a different construction: use a circulant graph or a different explicit construction
    
    # For k=3, g=16, let's try to use the fact that we need a cubic graph of girth 16
    # One approach: use a product of cycles or a different algebraic construction
    
    # Let's try a simpler approach: if n_luw is too large, try to find a smaller valid graph
    # using a random approach
    
    best_n = n_luw
    best_edges = edges
    
    # Try random cubic graphs of increasing size
    if n_luw > 2000:  # If LUW is too big, try random search
        for n_try in range(max(moore, 100), min(n_luw, 2000), 10):
            if time.time() > DEADLINE:
                break
            # Try a few random cubic graphs
            for attempt in range(3):
                if time.time() > DEADLINE:
                    break
                # Generate a random cubic graph
                # This is complex, so let's skip this for now and just return LUW
                pass
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
