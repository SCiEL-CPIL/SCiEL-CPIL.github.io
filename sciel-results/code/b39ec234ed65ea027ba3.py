"""Rewritten program using a randomized construction with girth enforcement."""
import numpy as np
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses a randomized approach with local search to find a valid graph.
    """
    # For (5,9), use a known good starting point or random construction
    # We'll use a random 5-regular graph and then try to increase girth
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Estimate a reasonable starting size
    # Moore bound for (5,9): 1 + 5*sum_{i=0}^{3} 4^i = 1 + 5*(1+4+16+64) = 1 + 5*85 = 426
    # Known best: 1116
    # Start with a size that's likely to work
    n_start = 1000
    
    # Try to find a valid graph
    best_n = None
    best_edges = None
    
    for n in range(n_start, n_start + 200, 10):
        # Generate random k-regular graph
        adj, edges = _random_regular_graph(n, k)
        if adj is None:
            continue
        
        # Check girth
        girth = _compute_girth(adj, n)
        
        if girth >= g:
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use a known construction
    return _fallback_construct(k, g)

def _random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices."""
    if n * k % 2 != 0:
        return None, None
    
    # Use configuration model
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    random.shuffle(stubs)
    
    adj = [[False] * n for _ in range(n)]
    edges = set()
    
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            continue  # Skip self-loops, will need to fix
        if adj[u][v]:
            continue  # Skip multi-edges
        adj[u][v] = adj[v][u] = True
        edges.add((min(u, v), max(u, v)))
    
    # Check if all vertices have degree k
    for i in range(n):
        if sum(adj[i]) != k:
            return None, None
    
    edge_list = list(edges)
    return adj, edge_list

def _compute_girth(adj, n):
    """Compute the girth of a graph using BFS from each vertex."""
    min_girth = float('inf')
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in range(n):
                if adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        
        # Find shortest cycle through start
        for v in range(n):
            if adj[start][v]:
                # Check if there's a cycle through start-v
                # This is not quite right; need to find shortest cycle
                pass
    
    # Better: use BFS to find shortest cycle
    # For each edge (u,v), find shortest path from u to v not using that edge
    # This is O(m*(n+m)) which is too slow for n=1000
    
    # Use a different approach: BFS from each vertex, track parent
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in range(n):
                if adj[u][v]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
    
    return min_girth if min_girth < float('inf') else n

def _fallback_construct(k, g):
    """Fallback to a known algebraic construction."""
    # Use a simple algebraic construction
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
# EVOLVE-BLOCK-END
