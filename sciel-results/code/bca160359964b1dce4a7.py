import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10 second buffer
    
    # 1. Generate LUW graph as fallback
    n_luw, edges_luw = _luw_construct(k, g)
    
    # 2. Try to find a smaller graph
    # Target N values to try
    # Moore bound for k=3, g=16 is 510.
    # Known best is 936.
    # Let's try N = 1000, 1500, 2000.
    target_ns = [1000, 1500, 2000, 5000]
    
    best_n = n_luw
    best_edges = edges_luw
    
    for n_target in target_ns:
        if time.time() > deadline:
            break
        # Try to find a valid graph of size n_target
        # We do this by generating random 3-regular graphs and checking girth.
        # If we find one, we update best.
        # We can try a few random instances.
        attempts = 10
        for _ in range(attempts):
            if time.time() > deadline:
                break
            edges = _generate_random_3_regular(n_target)
            if edges is None:
                continue
            if _has_girth_at_least(edges, n_target, g):
                if n_target < best_n:
                    best_n = n_target
                    best_edges = edges
                break # Found a valid one for this N, try smaller N? 
                      # Actually, we should try to find the smallest N.
                      # So we should iterate N in increasing order.
    
    return best_n, best_edges

def _luw_construct(k, g):
    # Copy from original
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

def _generate_random_3_regular(n):
    if n < 4 or n % 2 != 0:
        return None
    # Configuration model
    half_edges = [i for i in range(n) for _ in range(3)]
    random.shuffle(half_edges)
    edges = set()
    for i in range(0, 3*n, 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v:
            # Retry
            return _generate_random_3_regular(n)
        if (u, v) in edges or (v, u) in edges:
            return _generate_random_3_regular(n)
        edges.add((min(u, v), max(u, v)))
    return list(edges)

def _has_girth_at_least(edges, n, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for cycles of length < g
    # Use BFS from each edge? Or just check if any cycle < g exists.
    # A faster way: For each vertex, do BFS up to depth g//2 - 1.
    # If we find a back-edge to a vertex in the current path, we have a cycle.
    # This is O(n * (k^(g/2))) which is too slow for g=16, k=3.
    # 3^7 = 2187. n=1000. 2 million ops. Feasible.
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        queue = deque([(start, 0)])
        dist[start] = 0
        while queue:
            u, d = queue.popleft()
            if d >= g // 2 - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append((v, d + 1))
                elif v != parent[u]:
                    # Found a cycle
                    # Length is d + 1 + dist[v]
                    # But this might be a cycle that doesn't include start?
                    # Actually, BFS tree cycles are not necessarily the shortest.
                    # To be safe, we can just check if any cycle < g exists.
                    # A simpler check: 
                    # If we find a non-tree edge (u,v), the cycle length is dist[u] + dist[v] + 1.
                    # If dist[u] + dist[v] + 1 < g, return False.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
