"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def _check_girth(n, edges, min_girth):
    """Check if graph has girth at least min_girth using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS to find shortest cycle through start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _generate_random_k_regular(n, k, seed=None):
    """Generate a random k-regular graph on n vertices."""
    if seed is not None:
        random.seed(seed)
    
    # Each vertex needs k edges, total edges = n*k/2
    if n * k % 2 != 0:
        return None
    
    # Use configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    edges = []
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            continue
        if (u, v) in [(e[0], e[1]) for e in edges] or (v, u) in [(e[0], e[1]) for e in edges]:
            continue
        edges.append((min(u, v), max(u, v)))
    
    # Check if it's k-regular
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    return edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Moore bound: 1 + k * sum_{i=0}^{floor((g-1)/2)-1} (k-1)^i
    if g % 2 == 0:
        h = g // 2 - 1
    else:
        h = (g - 1) // 2 - 1
    
    moore_bound = 1 + k
    for i in range(1, h + 1):
        moore_bound += k * (k - 1) ** i
    
    # Try random search for a k-regular graph of girth g
    best_n = None
    best_edges = None
    
    # Start from a reasonable n and increase
    n_start = max(moore_bound, 200)
    n_end = 1000
    
    for n in range(n_start, min(n_end, 500), 10):
        if time.time() > deadline:
            break
        
        # Try multiple random graphs for this n
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            edges = _generate_random_k_regular(n, k, seed=hash((n, attempt)))
            if edges is None:
                continue
            
            if _check_girth(n, edges, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break  # Found a valid graph for this n, try smaller n
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use L-U-W construction
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
